# Attendance OCR Pipeline — Technical Documentation

This is the detailed reference for how the pipeline actually works internally. For install/setup steps, see [README.md](README.md) — this document covers architecture, the extraction algorithm, the verification and flagging systems, the output schema, and the non-obvious edge cases the code handles and why.

## 1. Overview

The pipeline takes a photographed or scanned handwritten attendance register (a single image or a multi-page PDF) and produces structured daily records: date, in-time, out-time, status (WORK/LEAVE/WOFF), and remark — plus a full validation/confidence layer flagging cells worth a human reviewing.

Two hosted APIs do all the "understanding" work; there is no local model server:

- **Mistral Document AI OCR** — reads each page image and produces the primary transcription (a markdown table) that everything else is built from.
- **Llama 4 Scout, via Groq** — an independent second opinion, only invoked on rows that already look suspicious, so it can catch a misread Mistral would never catch by checking its own work.

## 2. Pipeline flow

```
input file (image or PDF)
      |
      v
 [pdf_processor.py]  -- only if input is a PDF: renders each page to
      |                  an image via PyMuPDF, then hands each page
      |                  to the same per-page flow as a single image
      v
 [pipeline.py: split_double_page]
      |  Detects a photographed double-page spread (aspect ratio
      |  > 1.15, i.e. wider than tall) and splits it into left/right
      |  images BEFORE OCR ever sees them. An ordinary single page
      |  passes through unchanged.
      v
 [pipeline.py: process_page]  -- per (half-)page image:
      |
      |-- 1. crop_to_content()   (preprocessing/crop.py)
      |       Edge-based (Canny + morphological closing +
      |       quadrilateral polygon check) auto-crop to the page
      |       boundary, trimming desk/background clutter from a
      |       phone photo. No-ops (plain file copy, not a
      |       re-encode) if it can't confidently find a page-shaped
      |       contour -- never risks cropping into real content.
      |
      |-- 2. deskew_image()      (preprocessing/deskew.py)
      |       Canny + probabilistic Hough line detection finds the
      |       dominant near-horizontal line angle and rotates the
      |       image to straighten it -- but ONLY when that angle
      |       exceeds MIN_CORRECTION_DEGREES (3 degrees). Tried
      |       twice before with no threshold at all and removed both
      |       times: confirmed on a real page where the "skew" was
      |       just 0.91 degrees of measurement noise, yet rotating by
      |       even that much still corrupted column alignment
      |       (merged an "out" time into the adjacent signature
      |       column once, split a date across a phantom extra
      |       column another time). The threshold lets a genuinely
      |       tilted photo (5-15+ degrees from an unsteady hand)
      |       still get corrected while leaving an already-straight
      |       page alone. No-ops (plain file copy) below the
      |       threshold or if no lines are found.
      |
      |-- 3. MistralOCREngine.run()   (mistral_ocr_engine.py)
      |       The deskewed image goes to Mistral's OCR endpoint.
      |       If it comes back empty, retries once with the
      |       ORIGINAL untouched image -- confirmed empirically
      |       that preprocessing can occasionally make an otherwise
      |       legible page unreadable to Mistral even though it
      |       looks fine to the eye.
      |       Returns one or more "blocks" (see 3. below), each
      |       {"records": [...], "signature_name": "..."}.
      |
      |-- 4. Name resolution
      |       Page-level name (from text above the table) vs each
      |       block's own repeated "Sign" column text -- see 3.4.
      |
      |-- 5. process_records()   (post_processing.py)
      |       Normalizes each record's date/time/status strings into
      |       a consistent shape. Does NOT alter values otherwise --
      |       whatever Mistral read is what ends up in the output.
      |
      |-- 6. AttendanceValidator.validate()   (validators.py)
      |       Small rule-based cleanup: blanks times on LEAVE/WOFF
      |       rows, infers WORK/WOFF status when it's missing.
      |
      |-- 7. Row recheck   (llama_vision_engine.py + recheck_utils.py)
      |       For rows recheck_utils.needs_recheck() flags (invalid
      |       format, or a statistical outlier -- see 4. below),
      |       Llama 4 Scout independently re-reads the SAME page
      |       image and reports what it sees for that one row.
      |       Disagreement is recorded, not "corrected".
      |
      |-- 8. ValidationEngine.validate_records()   (validation_engine.py)
      |       Full flagging + confidence scoring pass -- see 5. below.
      |
      v
 [pipeline.py: _save_output]
      |  Writes outputs/<input-filename>.json and .xlsx
      |  (excel_report.py renders the human-readable spreadsheet view)
      v
   done
```

## 3. Extraction engine (`mistral_ocr_engine.py`)

Mistral's OCR returns markdown, not structured JSON — this module owns turning that markdown into records. This is the most heavily-evolved part of the codebase, because real handwritten registers vary wildly in column count, wording, and OCR rendering quirks.

### 3.1 Column-role classification, not fixed positions

Every header cell is matched against keyword patterns (`_ROLE_PATTERNS`) for `date`, `in_time`, `out_time`, `remark` — never by column index or exact wording. This is deliberate: two registers with different column counts, orders, or labels ("In time" vs "IN" vs fused "Intime") both work without any per-format special-casing.

When a header's own text is garbled or a header isn't legible at all, `_infer_missing_roles` falls back to guessing a role from the *shape* of a column's own data (does it look like `DD/MM/YY`? Does it look like `HH:MM`?) rather than trusting header text unconditionally. A majority of a column's non-blank values matching (`MIN_INFERENCE_MATCH_FRACTION = 0.7`), not every single one, is enough — real handwriting has enough noise that requiring a unanimous match would break on the first garbled cell.

### 3.2 Multi-block pages

A page can hold more than one full attendance table in two different shapes:
- **Side by side** in one wide table (`_find_group_ranges` — splits the header's columns into repeating groups, recognizable because "Date" appears more than once).
- **Sequentially**, one full table after another (`_find_headers` collects *every* header-shaped row, not just the first).

Each block is processed independently and becomes its own entry in the output.

### 3.3 Rejecting phantom blocks

A group that has a `date` column but never a single `in_time` or `out_time` column anywhere in its data is dropped entirely, rather than emitted as a fake employee with every time blank. This specifically catches a facing register page's edge bleeding into a photo's frame (its SNo/Date columns visible, but the shot cropped before that page's own In/Out columns) — a real attendance table always has at least one time column, so a group with none isn't one.

### 3.4 Employee name resolution

`extract_employee_name` looks for a markdown heading above the table first, then a plausible plain text line, then (last resort) name-like text merged into the first table row. Since the page-level name only ever identifies *one* employee, each block additionally checks its own repeated "Sign" column text (`_extract_block_signature_name`) — used instead of the page-level name only when it looks genuinely different (`pipeline._names_look_different`), which is how two different employees sharing one photographed page each get correctly labeled.

### 3.5 Handling Mistral response quirks

All confirmed against real captured responses, not hypothetical:

- **Bounding-box/caption JSON wrapper** — Mistral sometimes wraps the transcription in `[{"box_2d": ..., "caption": "...markdown..."}]` instead of returning plain markdown. `_unwrap_caption_json` extracts the caption text via regex (not `json.loads`) because the wrapper isn't always valid JSON to begin with.
- **HTML tables mixed with markdown** — Mistral occasionally renders one table on a page as markdown and a *different* table on the same page as raw `<tr><td>` HTML, sometimes glued directly onto the end of the previous markdown row with no newline at all. `_convert_html_tables` finds and rewrites any embedded HTML table into markdown in place, before the rest of parsing ever runs.
- **Repetition-loop decoding failure** — the decoder can get stuck endlessly repeating a fragment (e.g. an HTML tag) after correctly transcribing the real rows. `MAX_PLAUSIBLE_RECORDS_PER_BLOCK = 40` catches an implausibly large block (no real month has more rows than that) and discards the whole response, triggering the same fallback-to-original-image retry as a failed call.
- **Transient API failures** — `_call_ocr` retries up to `OCR_MAX_ATTEMPTS = 3` times with backoff on rate limits (429) and server errors (5xx), not on genuine bad requests. Confirmed empirically: pages that failed mid-batch-run succeeded instantly when retried individually right after.
- **Pipe-split dates** — a date like `01/05/26` occasionally comes back with its own separator rendered as a literal `|`, which collides with markdown's own cell delimiter and shreds it into 2-3 fragments (`_repair_pipe_split_date` merges them back).
- **Date+time merged in one cell** — some registers have the date and in-time written in the same physical box with no ruled line between them; Mistral transcribes it as one cell (`"01/05/26 14:00"`). `_infer_missing_roles` recognizes this shape and leaves `in_time` unassigned to a column, so `_rows_to_records` can split it back out of the date cell directly instead of forcing some unrelated column into the role.
- **Handwritten colon rendered as apostrophe** — `"6'.50"` for `6:50`. Tolerated as an optional extra character in the time regexes, and stripped outright in `post_processing.normalize_time`.
- **Fused header words** — `"Intime"`/`"Outtime"` with no space, which a plain `\bin\b`/`\bout\b` word-boundary match can't see. Explicit fused-word alternatives are in `_ROLE_PATTERNS`.

### 3.6 Struck-through rows

`_STRIKETHROUGH` detects markdown `~~text~~` — a partial signal only (Mistral doesn't always render a visibly crossed-out row this way). Checked only against a row's own core data (`date`, `in_time`, `out_time`), not every cell in its column range — a signature column's handwriting can have its own decorative stroke that isn't a cancellation mark, and checking the whole row would flag every row on a page where that happens.

## 4. Row recheck (`llama_vision_engine.py`, `recheck_utils.py`)

Not every cell gets rechecked — `recheck_utils.needs_recheck` gates it to only:
- a value that doesn't even look like a valid time format, or
- a statistical outlier (MAD-based, against the rest of that page's WORK rows' in/out times and shift durations) — `find_outlier_records`.

For a flagged row, `LlamaVisionEngine.recheck_row` sends the *same full page image* again (not a cropped cell — there's no per-cell bounding-box geometry available once Mistral's OCR API is the only engine) with a prompt naming the specific date, plus a disambiguator (block position and/or "the Nth row with this date") when that date repeats on the page.

`disagrees()` compares the two reads on full `HH:MM`. A disagreement doesn't overwrite Mistral's value — it's recorded as `{field}_llama_recheck` and `{field}_ocr_disagreement`, which feeds into the flagging system below.

## 5. Validation & flagging (`validation_engine.py`)

Full explanation of the three-stage design (evidence → decision → confidence), the complete list of checks, and — importantly — why a cell with a genuinely correct value sometimes still gets flagged, is in the project's own memory/prior conversation on this topic; the short version:

- **cell_highlight** is `CRITICAL evidence OR STRONG evidence` — see `CRITICAL_EVIDENCE`/`STRONG_EVIDENCE` in the file for the exact list (invalid format, missing value on a WORK row, impossible chronology, date sequence issues, struck-through, and Llama recheck disagreement).
- **attendance_analytics** (duration/time-of-day outliers) is a *separate* namespace that never influences highlighting — an unusual-but-real shift and a misread digit produce the same statistical signature, and conflating them previously caused legitimate shift changes to get flagged as OCR errors. It only decides which rows get a recheck in the first place.
- The system is deliberately tuned to over-flag: a genuinely correct cell that happens to look statistically unusual gets rechecked, and if Llama's independent read stumbles on that recheck, the disagreement flags a value that was actually right. This is accepted, not a bug — false positives are tolerable, silently missing a real misread is not.

## 6. Output JSON schema

**Single image**, or a PDF page/block that resolves to exactly one result:
```json
{
  "employee_name": "string",
  "total_records": 0,
  "records": [ /* see below */ ],
  "timings": { "Crop": 0.0, "Mistral OCR": 0.0, "Post Processing": 0.0,
               "Llama Row Recheck": 0.0, "Final Validation": 0.0, "Page Total": 0.0 }
}
```

**Multi-page PDF, or multi-block image** (more than one employee/table found):
```json
{
  "total_pages": 0,
  "total_employees": 0,
  "employees": [ { "page": 1, "employee_name": "...", "total_records": 0,
                   "records": [...], "timings": {...} }, ... ]
}
```

Each record -- a **clean row** (the common case; confirmed on a real page that ~86% of rows flag nothing at all) costs very little:
```json
{
  "date": "DD/MM/YY", "in_time": "HH:MM", "out_time": "HH:MM",
  "status": "WORK|LEAVE|WOFF|", "remark": "string", "struck_through": false,

  "in_time_llama_recheck": "HH:MM",   // only present if this cell was rechecked (real data, not a flag)

  "validation": [],              // list of only the checks that fired -- see names below
  "attendance_analytics": [],    // list of only the outliers that fired ("duration_outlier", "in_time_outlier", "out_time_outlier") -- never affects cell_highlight, see section 5
  "cell_highlight": [],          // list of only the flagged cells, e.g. ["out_time"] -- this IS the flagging decision
  "has_issue": false,
  "field_confidence": {}         // per-field entry omitted when it's a perfect, never-rechecked score -- present only for a field worth a reviewer's attention
}
```

A **flagged row** (real example, an `out_time_missing` case):
```json
{
  "date": "21/06/26", "in_time": "13:35", "out_time": "", "status": "WORK", "remark": "", "struck_through": false,
  "validation": ["out_time_missing"],
  "attendance_analytics": [],
  "cell_highlight": ["out_time"],
  "has_issue": true,
  "field_confidence": {
    "out_time": { "score": 50, "needs_review": true, "reasons": ["empty"] }
  }
}
```

`validation` names (only present in the list when that specific check fired): `date_invalid`, `in_time_invalid`, `out_time_invalid`, `in_time_missing`, `out_time_missing`, `chronology_invalid`, `time_alignment_suspect`, `status_time_mismatch`, `date_sequence_invalid`, `in_time_ocr_disagreement`, `out_time_ocr_disagreement`, `struck_through`.

Three fields that existed in earlier versions of this schema were removed as pure duplicates carrying no information beyond what's above: `row_highlight` (always identical to `has_issue`), `row_fully_invalid` (identical to `len(cell_highlight) == 4` -- all four cells flagged), and `verification` (a per-field `{"mistral": ..., "llama_recheck": ...}` view that never held anything not already directly on the record). None of this changes which cells get flagged or why -- only how compactly that same decision is written out.

## 7. Module reference

| File | Responsibility |
|---|---|
| `main.py` | Entry point — file picker (tkinter), dispatches to image or PDF flow |
| `pipeline.py` | Per-page orchestration: crop → OCR → name resolution → post-processing → recheck → validation → save |
| `pdf_processor.py` | Multi-page PDF driver — renders each page via PyMuPDF, calls into `pipeline.py` per page |
| `mistral_ocr_engine.py` | Mistral OCR call + all markdown-table parsing (see section 3) |
| `llama_vision_engine.py` | Llama 4 Scout (Groq) row recheck |
| `recheck_utils.py` | Engine-agnostic outlier detection and recheck gating |
| `post_processing.py` | Date/time/status string normalization (no forced month/year correction — output reflects exactly what was OCR'd) |
| `validators.py` | Small rule-based cleanup (blank times on off-days, infer missing status) |
| `validation_engine.py` | Full flagging + confidence scoring (see section 5) |
| `status_detector.py` | Fuzzy-matches LEAVE/WOFF keywords against a row's own text (handles OCR misspellings like "wloff", "1eave") |
| `preprocessing/crop.py` | Auto-crop a photographed page to its content boundary |
| `html_report.py` | Renders the `.html` report from the JSON structure |
| `config.py` | API keys (from `.env`), output path derivation (`get_output_paths`) |
| `constants.py` | Shared status-code vocabulary (`STATUS_WORK`/`STATUS_LEAVE`/`STATUS_WOFF`) |

## 8. Known limitations

- **Non-Latin script names** — name detection (`_HAS_LETTER` etc.) is Latin-script only; a Devanagari-written name won't be recognized as a name candidate.
- **Overwritten-but-not-struck-through cells** — only markdown strikethrough is caught; a genuinely overwritten/corrected digit that isn't rendered as strikethrough by Mistral has no dedicated detection (full-page recheck coverage on every cell would catch more of these, at roughly double the API cost per page — deliberately not done).
- **Recheck cross-contamination on duplicate dates** — when a date repeats on a page (an extra/overnight shift, or an undetected multi-block page), the recheck's disambiguator hint reduces but doesn't fully eliminate the chance Mistral reads back the wrong occurrence's value.
