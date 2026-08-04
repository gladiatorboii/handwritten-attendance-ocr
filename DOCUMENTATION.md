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
      |       The deskewed image goes to Mistral's OCR endpoint,
      |       requesting structured JSON directly (document_
      |       annotation_format) rather than a markdown table this
      |       class would otherwise have to parse itself -- see 3.
      |       below for why. If it comes back empty, retries once
      |       with the ORIGINAL untouched image -- confirmed
      |       empirically that preprocessing can occasionally make
      |       an otherwise legible page unreadable to Mistral even
      |       though it looks fine to the eye.
      |       Returns one or more "blocks" (almost always one; more
      |       than one when a page holds multiple employees), each
      |       {"records": [...], "signature_name": "...",
      |       "employee_code": "..."} -- name/code already resolved
      |       per employee directly by the model, no separate
      |       heading-scan step needed.
      |
      |-- 4. process_records()   (post_processing.py)
      |       Normalizes each record's date/time/status strings into
      |       a consistent shape. Does NOT alter values otherwise --
      |       whatever Mistral read is what ends up in the output.
      |
      |-- 5. AttendanceValidator.validate()   (validators.py)
      |       Small rule-based cleanup: blanks times on LEAVE/WOFF
      |       rows, infers WORK/WOFF status when it's missing.
      |
      |-- 6. Row recheck   (vision_recheck_engine.py + recheck_utils.py)
      |       For rows recheck_utils.needs_recheck() flags (invalid
      |       format, or a statistical outlier -- see 4. below),
      |       Qwen 3.6 27B (via Groq) independently re-reads the SAME
      |       page image and reports what it sees for those rows.
      |       Disagreement is recorded, not "corrected".
      |
      |-- 7. ValidationEngine.validate_records()   (validation_engine.py)
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

**Rewritten 2026-08-03** to request structured JSON directly from Mistral (`document_annotation_format`, an explicit JSON Schema — see `_EXTRACTION_SCHEMA`) instead of a markdown table transcription this module then had to parse itself. The markdown-parsing approach (column-role keyword matching, multi-block column-range splitting, header-value-shape inference, pipe-split-date repair, HTML/caption-JSON unwrapping, and a whole separate heading-text scanner for employee name/code) is gone — the model now returns each employee's name, code, and full list of daily records directly, matching a schema this module defines, guided by an explicit prompt (`_EXTRACTION_PROMPT`) describing the shapes a page can take (one employee; two sharing one wide table side by side; two sequential tables).

**Why the rewrite:** confirmed on a real page with two employees sharing one wide table (whose header was missing a serial-number cell for the second employee, shifting every later column label one position off from the data underneath it) that the old markdown parser lost the second employee's Out Time column entirely and got the first employee's In/Out direction backwards, with no way to tell from column shape alone which of two equally time-shaped columns was "in" vs "out". The header-text-driven approach had no way to recover from a header that mislabels its own columns. Asking the model for structured fields directly — with an explicit prompt describing the dual-table layout — got every row right on that same page, including direction, in direct side-by-side testing against the old approach.

**What this simplifies downstream:** `pipeline.py` no longer needs a separate page-level "base name" vs. per-block "signature name" reconciliation step (`_names_look_different`, the `"(table N)"` fallback label) — each block already carries its own `employee_name`/`employee_code` directly from the schema, since the model was asked to read each from that specific employee's own heading/signature text, not a page-wide title that might belong to the other employee.

### 3.1 What still runs after extraction

The schema only returns raw field values — the same post-extraction logic as before still applies to whatever comes back:
- **Status detection** (`status_detector.py`) — unchanged, still fuzzy-matches LEAVE/WOFF keywords (and truncated fragments like "WEE" or "WO") against each record's own text, checked only when no real time value is present (a row with real times is never overridden to Weekoff/Leave by stray text elsewhere).
- **Shift-code remark filtering** (`_SHIFT_CODE_REMARK`) — a remark is kept only if it's single-letter shift codes joined by `+` (e.g. `"A+B"`); anything else (`"M+Bag"`, free text) is dropped.
- **Time-value noise filtering** (`_is_noise_value`) — an in_time/out_time field that isn't actually a time (a WOFF/leave marker, or leftover letters once digits/separators are stripped) is blanked rather than kept as a bogus value.
- **Name/code light cleanup** (`_clean_employee_name`/`_clean_employee_code`) — a much smaller safety net than the old heading-scanner needed, since the model is now told exactly where to look rather than guessing among ambiguous candidates; still strips stray digit runs, `(Sub)`/trade-code tags from names, and validates a code is a bare 5-7 digit number before trusting it.
- **Padding-row / corrupted-response guard** — a fully-blank record (no date, times, or remark at all) past the real data is dropped silently; a block with implausibly many records (`MAX_PLAUSIBLE_RECORDS_PER_BLOCK = 40`) is treated as a corrupted response entirely, same as before.
- **Retry on transient failures** — `_call_ocr` still retries up to `OCR_MAX_ATTEMPTS = 3` times with backoff on rate limits (429) and server errors (5xx).

### 3.2 Struck-through rows

No longer detected via markdown strikethrough syntax (`~~text~~` can't survive being represented as a plain JSON string field) — the schema instead asks the model to report `struck_through` directly as a boolean per record, based on what it visually observes in the handwriting itself (a voided/crossed-out row), not a person's own signature stroke style.

## 4. Row recheck (`vision_recheck_engine.py`, `recheck_utils.py`)

Not every cell gets rechecked — `recheck_utils.needs_recheck` gates it to only:
- a value that doesn't even look like a valid time format, or
- a statistical outlier (MAD-based, against the rest of that page's WORK rows' in/out times and shift durations) — `find_outlier_records`.

Every flagged row on a page is batched into a *single* call to `VisionRecheckEngine.recheck_rows`, sending the same full page image (not a cropped cell — there's no per-cell bounding-box geometry available) alongside a list of every row that needs a second look, each with a date and a disambiguator (block position and/or "the Nth row with this date") when that date repeats on the page. Batching matters here specifically: the underlying model (Qwen 3.6 27B via Groq) has a small per-minute token quota on this account, and the page image itself (re-uploaded on every call) dominates that cost — one call per row per field burned through the whole budget after 1-2 calls on a real page; one call per page does not. A 429 is retried once after waiting out the window Groq itself reports.

`disagrees()` compares the two reads on full `HH:MM`. A disagreement doesn't overwrite Mistral's value — it's recorded as `{field}_recheck_value` and `{field}_ocr_disagreement`, which feeds into the flagging system below.

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
