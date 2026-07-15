import base64
import re
import time

import requests

from status_detector import StatusDetector
from config import debug_print

MISTRAL_OCR_URL = "https://api.mistral.ai/v1/ocr"

_status_detector = StatusDetector()

_TABLE_ROW = re.compile(r"^\s*\|(.+)\|\s*$")
_SEPARATOR_ROW = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")
_PLACEHOLDER_CELL = re.compile(r"^[\s\-‐-―_.*]*$")

# Words that mean "this cell isn't actually a time" (a WOFF/leave marker
# sitting in the in_time/out_time column instead of a real HH:MM value).
_STATUS_NOISE = {
    "woff", "w/off", "w off", "w.off", "w/oft", "wloff",
    "leave", "leaue", "leav", "holiday", "off", "weekly off",
    "week off", "weekoff", "w|off", "wooff",
}

# A real time cell, however it's formatted, is made up of digits plus
# separators/am-pm markers -- nothing else. Stripping all of that out and
# finding leftover letters means the cell actually contains a garbled
# status word (e.g. "W-0661", "WEEKDAY") rather than a time, even when it
# doesn't happen to match any of the known misspellings in _STATUS_NOISE.
# Includes "'" -- a handwritten colon sometimes gets OCR'd as an
# apostrophe right next to the real separator ("6'.50" for "6:50"), which
# would otherwise look like leftover non-time text on an actual time cell.
_TIME_RESIDUE = re.compile(r"\d+|[:.,\s\-']|a\.?m\.?|p\.?m\.?", re.IGNORECASE)


def _has_non_time_residue(text):
    return bool(_TIME_RESIDUE.sub("", text).strip())

# Lines that never denote the employee's name, even though they appear
# above/around the table header on most pages.
_NON_NAME_LINE = re.compile(
    r"^\s*(page\s*no\.?|date)\s*[:.]?\s*\d*\s*$"
    r"|^\s*[\W\d]*\s*$"                      # blank / punctuation-only / circled page numbers
    r"|^\s*(january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\s+\d{4}\s*$",
    re.IGNORECASE,
)

# An employee code is usually introduced by an explicit label right next
# to it ("emp id- 142810", "Emp: 153588", "empid 142810") -- this is an
# unambiguous signal, checked before any bare-number guess.
_EMP_CODE_LABELED = re.compile(
    r"emp\.?\s*(?:id|code|no)?\s*[:\-]\s*(\d{3,9})", re.IGNORECASE
)

# When no label exists at all, the code is just a bare number sitting in
# the heading next to the name (e.g. "(Sub) PINKU CHOUDRAY 14396") --
# same heading text also carries a phone number, a book/register
# reference number, the register's own "<Month> <year>" line (e.g. "June
# 2026" -- confirmed on a real page where the bare-number fallback
# mistook the plain "2026" for a code), and the employee's own circled
# serial number, none of which are the code. An employee code observed
# on real registers this project has seen is always 5-6 digits; a floor
# of 5 (not 4) rules out a 4-digit calendar year without needing to
# recognize a year as such, and capping at 9 digits rules out a 10+
# digit phone number the same way. The 2-digit serial number already
# falls below the floor on its own.
_EMP_CODE_BARE = re.compile(r"\b\d{5,9}\b")

# A book/register reference number (e.g. "doob No 9A/1120925") can be the
# same length as a real employee code -- confirmed on a real page where
# this sits in the same heading as a genuine (labeled) employee code.
# Skipped outright during the bare-number fallback so it's never
# mistaken for one on some other page that lacks a label entirely.
_NON_CODE_LINE = re.compile(r"\b(?:book|doob|regd?)\.?\s*no\b", re.IGNORECASE)

# Column roles are found by matching each header cell's own text against
# these keywords, not by assuming a fixed column count/order/exact
# wording -- different registers label (and order, and count) their
# columns differently (e.g. "Date | In time | Sign | Out time | Sign"
# vs "SNo | Date | IN | sig | OUT | sig | Rem"), so the table's own
# header row is the only thing that can tell us where date/in/out live.
_ROLE_PATTERNS = {
    "date": re.compile(r"date", re.IGNORECASE),
    # "outtime"/"intime" (no space) show up when a header cell's two
    # words get transcribed fused together -- \bout\b/\bin\b alone
    # require a word boundary on both sides, which a fused word doesn't
    # have (nothing separates "out" from the "time" right after it).
    "out_time": re.compile(r"\bout\b|time\s*out|outtime|check\s*-?\s*out|log\s*-?\s*out", re.IGNORECASE),
    "in_time": re.compile(r"\bin\b|time\s*in|intime|check\s*-?\s*in|log\s*-?\s*in", re.IGNORECASE),
    # "Remark"/"Remand"/"Action" columns carry an overtime / extra-shift
    # code (e.g. "M+B", "M.T Bag") -- "re" alone shows up when OCR
    # truncates "Remark" to just its first two letters.
    "remark": re.compile(r"\brem\w*|\baction\b|\bovertime\b|\bo\.?t\.?\b|^re$", re.IGNORECASE),
}

# Fallback signal when a header cell's own text is unreadable (e.g. OCR
# garbled "Date" into "DETC", or "OUT" into "OCT") -- a column's actual
# values still look like a date/time regardless of what its header says,
# so the data itself can locate it (see _infer_missing_roles).
_DATE_VALUE_PATTERN = re.compile(r"^\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}$")

# A handwritten colon sometimes gets OCR'd as an apostrophe immediately
# before the real separator ("6'.50" for "6:50") rather than being
# recognized as a colon outright -- tolerated here as an optional extra
# character rather than the separator itself, alongside the "." Mistral
# more commonly substitutes for ":".
_TIME_VALUE_PATTERN = re.compile(r"^\d{1,2}'?[:.]\d{2}\s*(a\.?m\.?|p\.?m\.?)?$", re.IGNORECASE)
_VALUE_PATTERNS = {
    "date": _DATE_VALUE_PATTERN,
    "in_time": _TIME_VALUE_PATTERN,
    "out_time": _TIME_VALUE_PATTERN,
}

# A looser "is this even roughly a time" check than _TIME_VALUE_PATTERN --
# real handwriting/OCR routinely drops the minute's leading zero ("17:5"
# for "17:05"), which the strict pattern rejects. Used only to confirm an
# already header-claimed in_time/out_time role still belongs where the
# header put it (see _infer_missing_roles): a real time column with a few
# single-digit minutes shouldn't get its header-given role revoked, even
# though that same looseness would be too permissive for *claiming* a
# column nobody has identified yet (where _TIME_VALUE_PATTERN's stricter
# match is what keeps a non-time column from being mistaken for one).
_TIME_LOOSE_PATTERN = re.compile(r"^\d{1,2}'?[:.]\d{1,2}\s*(a\.?m\.?|p\.?m\.?)?$", re.IGNORECASE)

# Some registers write the date and the in-time in the same physical box
# (no ruled line between them), which Mistral then transcribes as one
# cell, e.g. "01/05/26 14:00" -- rather than two cells split across
# separate columns like every other row's date/time. See
# _infer_missing_roles / _rows_to_records for how this is detected and
# split back into its two values.
_DATE_TIME_MERGED_PATTERN = re.compile(
    r"^(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\s+(\d{1,2}[:.]\d{2})$"
)

# A short bare numeric fragment, the shape a date's own day/month/year
# piece takes once a scan's "01|05|26" style date has been shredded by
# the table parser's own "|" cell delimiter (see _repair_pipe_split_date).
_NUMERIC_FRAGMENT = re.compile(r"^\d{1,4}$")

# Mistral sometimes represents a visibly crossed-out/struck-through cell
# using markdown strikethrough syntax in its transcription. This is not
# a reliable signal -- confirmed empirically that the same kind of
# crossed-out row is sometimes transcribed as plain text instead, with
# no marker at all -- so this only catches the clearer cases Mistral
# itself noticed, not every struck-through cell in the source.
_STRIKETHROUGH = re.compile(r"~~.+?~~")

# See MistralOCREngine._unwrap_caption_json: extracts each "caption"
# value out of a bounding-box/caption JSON wrapper without requiring the
# wrapper itself to be valid JSON.
_CAPTION_PATTERN = re.compile(r'"caption"\s*:\s*"(.*?)"\s*\}', re.DOTALL)


def _classify_header(cells):
    """Column role -> index, by matching each header cell's text against
    _ROLE_PATTERNS. Only the first cell matching a given role counts, so
    "Sign"/"SNo"-style columns (matching none of these patterns) are
    simply left unmapped rather than needing to be named explicitly."""
    roles = {}
    for idx, cell in enumerate(cells):
        text = cell.strip()
        if not text:
            continue
        if "date" not in roles and _ROLE_PATTERNS["date"].search(text):
            roles["date"] = idx
        elif "out_time" not in roles and _ROLE_PATTERNS["out_time"].search(text):
            roles["out_time"] = idx
        elif "in_time" not in roles and _ROLE_PATTERNS["in_time"].search(text):
            roles["in_time"] = idx
        elif "remark" not in roles and _ROLE_PATTERNS["remark"].search(text):
            roles["remark"] = idx
    return roles


def _find_group_ranges(cells):
    """
    Splits a header row's columns into one or more (start, end) column
    ranges, each range covering one attendance-table "block". Most
    registers have exactly one block; some (e.g. two employees, or two
    shifts, sharing one physical page) repeat the whole column set
    side by side in one wide table -- recognizable because "Date" (the
    one column every block reliably has) appears more than once.

    Each block starts as many columns before its own "Date" cell as the
    *first* block's "Date" cell sits after column 0 -- not exactly at its
    "Date" position -- since a leading serial-number column (if present)
    repeats before every block's "Date", not just the first one.
    """
    date_positions = [
        i for i, c in enumerate(cells) if c.strip() and _ROLE_PATTERNS["date"].search(c.strip())
    ]
    if len(date_positions) <= 1:
        return [(0, len(cells))]

    offset = date_positions[0]
    starts = [max(0, pos - offset) for pos in date_positions]
    for i in range(1, len(starts)):
        if starts[i] <= starts[i - 1]:
            starts[i] = starts[i - 1] + 1

    ranges = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(cells)
        ranges.append((start, end))

    # The boundary above assumes every block repeats the same column
    # width -- true for two genuinely symmetric blocks, but false when a
    # later block is a photo-frame fragment that stops before its own
    # In/Out/Sig columns exist at all (its "Date" then sits closer to
    # the previous block's Date than a full block-width away). When that
    # happens, the earlier block's own trailing time/remark column(s)
    # get sliced into the next block's range instead of staying in the
    # one they actually belong to. Reclaim any such cell for the earlier
    # block: as long as the next block's leading cell isn't itself a
    # "Date" (the genuine start of that block) and it fills a role the
    # earlier block doesn't have yet, it belongs there instead.
    for i in range(len(ranges) - 1):
        start, end = ranges[i]
        next_start, next_end = ranges[i + 1]
        while next_start < next_end:
            candidate = cells[next_start].strip()
            if not candidate or _ROLE_PATTERNS["date"].search(candidate):
                break
            current_roles = _classify_header(cells[start:end])
            if not any(
                role not in current_roles and _ROLE_PATTERNS[role].search(candidate)
                for role in ("out_time", "in_time", "remark")
            ):
                break
            end += 1
            next_start += 1
        ranges[i] = (start, end)
        ranges[i + 1] = (next_start, next_end)

    return ranges

_HAS_LETTER = re.compile(r"[A-Za-z]")

# Vision LLMs occasionally get stuck in a decoding loop on a hard/
# ambiguous handwritten region, repeating the same short token many times
# instead of producing real text (e.g. "SUP: P. P. P. P. P. ..."). This
# is a known general failure mode, not specific to any one document --
# confirmed non-deterministic here: re-sending the same page produced a
# clean read, so treat a name candidate exhibiting it as unusable rather
# than trying to salvage a "real" name out of the repeated tokens.
_REPEATED_TOKEN = re.compile(r"\b(\S{1,4})\b(?:[\s.,]+\1\b){3,}", re.IGNORECASE)

# On at least one hard page, Mistral's response body wasn't a name/table
# transcription at all but a stray vision-model layout annotation (e.g.
# '[{"box_2d": [0, 0, 998, 998], "label": "table", "caption": "<table>').
# That text has letters and no repeated token, so it otherwise passed as
# a plausible name. No real employee name ever contains JSON's structural
# characters, so reject any candidate that does.
_NAME_DISALLOWED_CHARS = re.compile(r'[{}\[\]"]')

# A real employee name never contains a run of digits -- a phone number,
# employee code, or serial number merged into the same cell as the name
# (confirmed on a real page: "SUP. Pushpendra : 8426035466") is always
# noise, not part of the name. Floor of 3 so a single stray digit from a
# misread letter doesn't nuke an otherwise-fine candidate.
_DIGIT_RUN = re.compile(r"\d{3,}")

# The "Page No." / "Date" mini-table's own column labels, when they get
# merged into the same line as the name instead of staying on their own
# row (confirmed on a real page: "(90) Deep Sing. Page No. Date") -- a
# real name never contains either literally.
_LABEL_NOISE = re.compile(r"\bpage\s*no\.?\b|\bdate\b", re.IGNORECASE)

# Left behind once a parenthesized serial number's digits are stripped
# out (e.g. "(77)" -> "()") -- never meaningful on its own.
_EMPTY_PARENS = re.compile(r"\(\s*\)")


def _strip_numeric_noise(text):
    cleaned = _DIGIT_RUN.sub("", text)
    cleaned = _LABEL_NOISE.sub("", cleaned)
    cleaned = _EMPTY_PARENS.sub("", cleaned)
    cleaned = re.sub(r"^[\s:./\-]+|[\s:./\-]+$", "", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()

# Circled/dingbat digit glyphs Mistral sometimes renders a page number as
# (e.g. "⑦"), plus plain digits -- stripped from the front of a name
# candidate since these count as Unicode word characters (so a plain
# [\W\d]* filter doesn't catch a name candidate like "⑦ RulbIR").
_LEADING_GLYPH = re.compile(r"^[①-⑳⓪❶-➓\d\s]+")


def _strip_leading_glyph(text):
    return _LEADING_GLYPH.sub("", text).strip()


def _looks_like_name(text):
    return (
        bool(_HAS_LETTER.search(text))
        and not _REPEATED_TOKEN.search(text)
        and not _NAME_DISALLOWED_CHARS.search(text)
    )


def _is_status_noise(text):
    """True if a cell's text is a status word, not a time/date."""
    t = text.lower().strip().replace("/", " ").replace(".", " ")
    return any(noise in t for noise in _STATUS_NOISE)


def _is_expected_time_placeholder(text):
    """
    True if a blank in_time/out_time cell is *expected* to be blank -- a
    recognizable status word (WOFF/leave/off) or a bare placeholder mark
    (blank, dash). Narrower than _is_noise_cell: used when judging
    whether a column's values look like times at all (see
    _infer_missing_roles), where a legitimate "no time today" marker
    shouldn't count as evidence either way, but genuinely wrong content
    (e.g. a signature name sitting in what should be a time column)
    should still count as evidence the column is misidentified --
    _is_noise_cell's broader "any leftover letters" check would wrongly
    swallow that signal too.
    """
    return _is_status_noise(text) or bool(_PLACEHOLDER_CELL.match(text))


def _is_noise_cell(text):
    """
    True if an in_time/out_time cell isn't actually a time -- either a
    recognizable status word/placeholder, or (more generally) any text
    that has leftover letters once every digit/separator/am-pm token is
    stripped out, which is what a status word garbled during OCR looks
    like even when it doesn't match a known misspelling.
    """
    return (
        _is_expected_time_placeholder(text)
        or _has_non_time_residue(text)
    )


class MistralOCREngine:
    """
    Full-page OCR engine for the pipeline: Mistral's hosted Document AI
    OCR endpoint provides the page read (records + employee name). Surya/
    EasyOCR were removed from the pipeline entirely (see git history on
    llama-vision-integration/main for that version) once benchmarking
    showed Mistral's date/time reads were reliable enough to stand alone
    as the primary engine -- this class does not fall back to any other
    engine for the initial read.

    Talks to Mistral's REST API directly with `requests` rather than the
    `mistralai` SDK: the SDK pins httpx>=0.28.1, which conflicted with
    surya-ocr's httpx<0.28 requirement when both lived in the same venv
    (confirmed while building the standalone test in mistral_ocr_test/).
    Now that Surya is gone this conflict can't recur, but there's no
    reason to add the SDK's dependency footprint back for a single
    REST call this class already makes directly.

    Mistral's OCR returns a markdown transcription of the page, not
    structured per-field JSON -- so `run()` parses the markdown table
    into records itself, reusing StatusDetector so a WOFF/LEAVE cell is
    classified the same way regardless of which line it came from.

    Does NOT recheck its own flagged rows -- see llama_vision_engine.py
    for why that's a genuinely different model (Llama 4 Scout via Groq)
    instead of Mistral re-reading itself, and recheck_utils.py for the
    engine-agnostic outlier/format gate deciding which rows get
    rechecked at all.
    """

    # See _infer_missing_roles: a column counts as "this role's kind of
    # value" once this fraction of its probed non-blank cells match,
    # rather than requiring every single one to.
    MIN_INFERENCE_MATCH_FRACTION = 0.7

    # See run(): a single block with more rows than this is treated as a
    # corrupted/repetition-loop response rather than genuine data -- a
    # month's register realistically tops out around 31-35 rows even
    # with a few duplicate-date extra shifts.
    MAX_PLAUSIBLE_RECORDS_PER_BLOCK = 40

    # See _call_ocr: a sustained batch run (many pages back-to-back) can
    # hit a transient failure -- a rate limit, a dropped connection, a
    # momentary 5xx from Mistral's side -- that has nothing to do with
    # the page itself and clears up if retried a moment later (confirmed
    # empirically: pages that failed mid-batch succeeded immediately when
    # retried individually right after). Without a retry, one such blip
    # silently turns into a blank page in the final output.
    OCR_MAX_ATTEMPTS = 3
    OCR_RETRY_DELAY_SECONDS = 2

    def __init__(self, api_key, model="mistral-ocr-latest"):
        self.api_key = api_key
        self.model = model
        self.last_markdown = ""

    # ------------------------------------------------------------------
    # FULL-PAGE OCR
    # ------------------------------------------------------------------
    def run(self, image_path):
        """
        Returns a list of "blocks" -- almost always exactly one, but more
        than one when a page holds multiple full attendance tables (two
        employees, or two shifts, sharing one physical page -- either
        side by side in one wide table, or one after another as fully
        separate tables; see _find_group_ranges / _find_headers). Each
        block is {"records": [...], "signature_name": "..."} -- records
        is a list of record dicts (date, in_time, out_time, status,
        remark); signature_name is whichever name-like text repeats most
        in that block's own rows (its Sign column, usually), "" if
        nothing clears that bar. Needed because the page-level name (see
        extract_employee_name) only ever identifies one employee, which
        is wrong for the *other* block on a page shared by two different
        people.

        Returns [] if the call fails or no usable table is found at all.
        Also stores the raw markdown on self.last_markdown for
        extract_employee_name.
        """
        self.last_markdown = ""
        try:
            markdown = self._call_ocr(image_path)
            self.last_markdown = markdown
            blocks = self._parse_markdown_table(markdown)

            # A real month's attendance register has at most a few dozen
            # rows -- confirmed on a real page that Mistral's decoder can
            # get stuck in a repetition loop (endlessly repeating an
            # HTML table fragment) after transcribing the real rows
            # correctly, which parses as one block with the real ~20
            # rows followed by hundreds of blank/garbage ones. Rather
            # than silently returning a block flooded with junk records,
            # treat an implausibly large block as evidence the whole
            # response was corrupted and let the caller's existing
            # empty-result handling (retry with a different image) take
            # over instead.
            if any(len(b["records"]) > self.MAX_PLAUSIBLE_RECORDS_PER_BLOCK for b in blocks):
                debug_print(
                    "MistralOCREngine: implausibly large block "
                    "(likely a repetition-loop response), discarding"
                )
                return []

            return blocks
        except Exception as e:
            debug_print(f"MistralOCREngine: full-page call failed: {e}")
            return []

    def extract_employee_name(self, markdown):
        """
        The employee name is plain text above the table on most pages
        (e.g. "Anand", "Sujit Sankar."), but on at least one observed page
        it was misread straight into the first row of a mini-table above
        the real header instead of appearing as a separate text line --
        so this checks both places.

        A markdown heading ("# Name") is checked before any plain line:
        one page had "Electrical" (a department label, plain text) above
        the real "# Deep Sing." heading, and the first-plausible-line
        approach picked the department label instead. A heading is a much
        stronger signal when one exists at all; falling back to the first
        plausible plain line only when no heading is present doesn't
        regress pages that never have a heading in the first place.
        """
        if not markdown:
            return ""

        lines = markdown.splitlines()
        table_start = next((i for i, l in enumerate(lines) if _TABLE_ROW.match(l)), len(lines))
        pre_table_lines = lines[:table_start]

        for line in pre_table_lines:
            stripped = line.strip()
            if not stripped.startswith("#"):
                continue
            candidate = _strip_leading_glyph(stripped.lstrip("#").strip())
            if candidate and _looks_like_name(candidate) and not _NON_NAME_LINE.match(candidate):
                return _strip_numeric_noise(candidate.rstrip("."))

        for line in pre_table_lines:
            candidate = _strip_leading_glyph(line.strip().lstrip("#").strip())
            if candidate and _looks_like_name(candidate) and not _NON_NAME_LINE.match(candidate):
                return _strip_numeric_noise(candidate.rstrip("."))

        # Fallback: name text merged into the first table row -- in any
        # cell, not just the first (an employee code/serial number
        # sometimes occupies the first cell with the actual name text
        # sitting in a later one instead).
        table_lines = [l for l in lines if _TABLE_ROW.match(l)]
        if table_lines:
            cells = [c.strip() for c in table_lines[0].strip().strip("|").split("|")]

            # If this first table-row line is actually the real column
            # header (some pages have no separate name line/mini-table at
            # all, so the header is the very first "|" row) -- its cells
            # are column labels like "SNo"/"Date"/"IN", not name text, and
            # would otherwise get mistaken for one (e.g. "SNo" itself
            # passing every name-shaped check).
            header_roles = _classify_header(cells)
            is_header_row = "date" in header_roles and (
                "in_time" in header_roles or "out_time" in header_roles
            )

            if not is_header_row:
                for cell in cells:
                    candidate = _strip_leading_glyph(cell)
                    if candidate and _looks_like_name(candidate) and not _NON_NAME_LINE.match(candidate):
                        return _strip_numeric_noise(candidate.rstrip("."))

        return ""

    def extract_employee_code(self, markdown):
        """
        The employee code sits in the same heading area as the name (see
        extract_employee_name), usually right next to a phone number, a
        book/register reference number, and/or the employee's own
        circled serial number -- none of which are the code. Also checks
        every mini-table row above the real column-header row, not just
        plain pre-table text: on at least one observed page the whole
        heading (name, code, phone) got merged into that mini-table
        instead of appearing as separate text lines, the same place
        extract_employee_name falls back to.
        """
        if not markdown:
            return ""

        lines = markdown.splitlines()
        table_start = next((i for i, l in enumerate(lines) if _TABLE_ROW.match(l)), len(lines))
        pre_table_lines = lines[:table_start]

        code = self._find_code_in_lines(pre_table_lines)
        if code:
            return code

        table_lines = [l for l in lines if _TABLE_ROW.match(l)]
        return self._find_code_in_lines(self._mini_table_lines(table_lines))

    def _mini_table_lines(self, table_lines):
        """
        Rows above the real column-header row -- the boundary is
        whichever row's cells are the first to look like actual column
        labels (a "date" role plus an "in"/"out" role), same signal
        extract_employee_name uses to tell a mini-table row from the
        real header.
        """
        for i, line in enumerate(table_lines):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            roles = _classify_header(cells)
            if "date" in roles and ("in_time" in roles or "out_time" in roles):
                return table_lines[:i]
        return table_lines

    def _find_code_in_lines(self, candidate_lines):
        labeled = _EMP_CODE_LABELED.search("\n".join(candidate_lines))
        if labeled:
            return labeled.group(1)

        for line in candidate_lines:
            if _NON_CODE_LINE.search(line):
                continue
            bare = _EMP_CODE_BARE.search(line)
            if bare:
                return bare.group(0)

        return ""

    # ------------------------------------------------------------------
    # MARKDOWN TABLE PARSING
    # ------------------------------------------------------------------
    def _parse_markdown_table(self, markdown):
        """
        Finds every attendance-table header row among all markdown table
        rows on the page (Mistral often also emits small "Page No." /
        "Date" mini-tables immediately before the real one, sometimes
        with no blank line separating them, so grouping by blank-line-
        delimited blocks isn't reliable) -- not just the first, since a
        page can hold two full tables one after another (e.g. two
        different employees whose physical register pages were
        photographed together), each with its own legible header. Each
        header row can also itself hold more than one column-group side
        by side (see _find_group_ranges -- two employees, or two shifts,
        sharing one wide table instead of two sequential ones). Neither
        the header string, column count, nor number of tables/groups is
        fixed, since different registers vary all of them.
        """
        lines = markdown.splitlines()
        table_lines = [line for line in lines if _TABLE_ROW.match(line)]

        headers = self._find_headers(table_lines)

        if not headers:
            # Last resort: no row's header text was legible enough to
            # find any date/in/out signal at all (Mistral occasionally
            # misplaces its own header/separator markdown so the table
            # starts straight into data). Try inferring a single implicit
            # group purely from the shape of every row's values.
            non_separator_lines = [l for l in table_lines if not _SEPARATOR_ROW.match(l)]
            probe_rows = self._split_rows(non_separator_lines)
            # No cap here (unlike the per-group inference below): at this
            # point nothing about the table is known yet -- no header,
            # no column boundaries -- so there's no other row range to
            # scope the probe to. A page with more leading noise lines
            # before its data starts (e.g. a title line plus a "Page No."
            # mini-table plus the real header, none of which ever got
            # recognized as a header) would otherwise dilute a small
            # fixed-size window below the match-fraction threshold and
            # lose the whole page, even though the real data rows
            # comfortably outnumber the noise once all of them are seen.
            inferred = self._infer_missing_roles(
                {}, probe_rows, max_probe_rows=len(probe_rows)
            )
            if "date" not in inferred:
                return []

            # Since no row was ever recognized as a header, whatever sits
            # above the real data (an employee name/phone-number line, or
            # the header row itself, unrecognized because it uses wording
            # like "Orig"/"Orig" instead of "In"/"Out") is still sitting
            # in table_lines undistinguished from genuine data rows.
            # Skip ahead to wherever the inferred date column actually
            # starts looking like a real date, the same way a normal
            # header_idx would exclude everything above it.
            date_idx = inferred["date"]
            header_idx = -1
            for i, line in enumerate(table_lines):
                if _SEPARATOR_ROW.match(line):
                    continue
                cells = self._split_rows([line])[0]
                value = cells[date_idx].strip() if date_idx < len(cells) else ""
                if _DATE_VALUE_PATTERN.match(value):
                    header_idx = i - 1
                    break

            width = max((len(r) for r in probe_rows), default=0)
            headers = [(header_idx, [((0, width), inferred)])]

        blocks = []

        for i, (header_idx, groups) in enumerate(headers):
            next_header_idx = headers[i + 1][0] if i + 1 < len(headers) else len(table_lines)
            data_lines = [
                row for row in table_lines[header_idx + 1:next_header_idx]
                if not _SEPARATOR_ROW.match(row)
            ]
            probe_rows = self._split_rows(data_lines)

            for col_range, roles in groups:
                # Repaired the same way _rows_to_records repairs each real
                # row -- an unrepaired pipe-split date ("01|05|26" -> 3
                # cells instead of 1) shifts every column after it two
                # positions late, so probing the raw split would validate
                # in_time/out_time against date fragments instead of the
                # real columns the header actually points at.
                repaired_probe_rows = [
                    self._repair_pipe_split_date(row, roles.get("date"))
                    for row in probe_rows
                ]

                # Whichever roles a group's own header text didn't cover
                # (garbled "Date" -> "DETC", "OUT" -> "OCT", etc.) get a
                # second chance inferred from the shape of that group's
                # own data segment.
                roles = self._infer_missing_roles(roles, repaired_probe_rows, col_range)
                if "date" not in roles:
                    continue

                # A group with a date column but never a single in_time
                # or out_time column, anywhere in its own data, isn't a
                # genuine second table -- a real attendance table always
                # has at least one time column. This shape instead
                # matches a facing register page's edge bleeding into the
                # photo's frame (its SNo/Date columns visible, but the
                # shot is cropped before reaching that page's own In/
                # Out/Sig columns). Emitting it as its own block would
                # fabricate a phantom employee entry with every time
                # blank, out of content that was never meant to be read
                # as a complete table on its own.
                if "in_time" not in roles and "out_time" not in roles:
                    continue

                blocks.append({
                    "records": self._rows_to_records(data_lines, roles, col_range),
                    "signature_name": self._extract_block_signature_name(
                        data_lines, roles, col_range
                    ),
                })

        return blocks

    def _extract_block_signature_name(self, data_lines, roles, col_range, min_fraction=0.4):
        """
        The page-level name (extract_employee_name) only ever identifies
        one employee, but when +++++a page holds more than one table (e.g. two
        different employees whose physical register pages were
        photographed together, see page3 in early testing -- one table's
        rows all said "Pushpenna" in the sign column while the page's
        only visible name text was the *other* employee's), each block's
        own rows usually repeat that employee's own signature/initials in
        a column this method didn't otherwise claim. Returns whichever
        not-yet-claimed column's text is most common across this block's
        rows, if it shows up in a real majority of them and looks like a
        name rather than a checkmark/status word -- "" if nothing clears
        that bar, so callers can fall back to the page-level name.
        """
        claimed = {i for i in roles.values() if i is not None}
        start, end = col_range
        counts = {}

        for row in data_lines:
            cells = [c.strip() for c in row.strip().strip("|").split("|")]
            for idx in range(start, min(end, len(cells))):
                if idx in claimed:
                    continue
                text = cells[idx]
                if len(text) < 3 or not _looks_like_name(text) or _is_status_noise(text):
                    continue
                counts[text] = counts.get(text, 0) + 1

        if not counts:
            return ""

        name, count = max(counts.items(), key=lambda kv: kv[1])
        if count < max(2, len(data_lines) * min_fraction):
            return ""
        return name

    def _find_headers(self, table_lines):
        """
        Returns a list of (header_idx, groups) for every row that
        classifies as having, in at least one column-group, both a date
        column and an in/out time column -- there can be more than one
        such row if the page holds multiple full tables in sequence.

        Falls back to the single best partial (in/out-only) match only
        if *no* row anywhere has a full date+time match -- a lone
        in_time/out_time-shaped header cell isn't trusted as a table
        boundary on its own (a data row could coincidentally brush a
        pattern), so it's only used as a last resort, not stacked with
        genuine full-match headers found elsewhere on the same page.
        """
        full_headers = []
        fallback_idx = fallback_groups = None

        for i, line in enumerate(table_lines):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            groups = [
                (col_range, _classify_header(cells[col_range[0]:col_range[1]]))
                for col_range in _find_group_ranges(cells)
            ]
            # Offset each group's role indices back to absolute column
            # positions (_classify_header only sees its own slice).
            groups = [
                (col_range, {role: idx + col_range[0] for role, idx in roles.items()})
                for col_range, roles in groups
            ]

            has_full_group = any(
                "date" in roles and ("in_time" in roles or "out_time" in roles)
                for _, roles in groups
            )
            has_partial_group = any(
                "in_time" in roles or "out_time" in roles for _, roles in groups
            )

            if has_full_group:
                full_headers.append((i, groups))
            elif fallback_idx is None and has_partial_group:
                fallback_idx, fallback_groups = i, groups

        if full_headers:
            return full_headers

        if fallback_idx is not None:
            return [(fallback_idx, fallback_groups)]

        return []

    def _split_rows(self, rows):
        return [[c.strip() for c in row.strip().strip("|").split("|")] for row in rows]

    def _infer_missing_roles(self, roles, probe_rows, col_range=None, max_probe_rows=12):
        """
        Fallback for when a header cell's own text for date/in_time/
        out_time came back garbled from OCR (e.g. "Date" -> "DETC",
        "OUT" -> "OCT") but the row was still identifiable as the header
        via its other columns -- or when no header was legible at all.
        Looks at the next few data rows and, for whichever roles are
        still missing, claims whichever not-yet-used column (within
        col_range, if given -- so one block's inference can't reach into
        a neighboring block's columns) whose non-blank values consistently
        look like that role's kind of value, rather than relying on
        header text at all.

        Also drops an already-assigned in_time/out_time role if that
        column's own values don't actually look like that role's kind of
        value -- a header cell can match a role's keyword yet still be
        positioned wrong (e.g. one header cell's text absorbing a
        neighboring column's label, shifting every later header label one
        column early relative to the data it's supposed to describe). A
        dropped role gets re-derived below the same way a genuinely
        missing one would be, from whichever unclaimed column actually
        fits, instead of leaving garbage parked in the role.

        Also recognizes a date column whose cells hold a date immediately
        followed by a time in the same cell (the date and in-time were
        written in one box in the source register, e.g. "01/05/26
        14:00", not two separate columns) -- when in_time isn't
        otherwise findable as its own column, this leaves it unassigned
        here rather than forcing some unrelated column into the role, so
        _rows_to_records can split it out of that same date cell instead.
        """
        rows = probe_rows[:max_probe_rows] if probe_rows else []
        max_cols = max((len(r) for r in rows), default=0)
        start, end = col_range if col_range else (0, max_cols)

        roles = dict(roles)

        for role in ("in_time", "out_time"):
            col = roles.get(role)
            if col is None:
                continue
            # A leave/WOFF row's placeholder ("--", "off", etc) is a
            # legitimate "no time expected here" marker, not a value that
            # should look like a time -- counting it as one would dilute
            # a genuinely correct column's match fraction on any page
            # with enough leave/WOFF rows in the probe window, and
            # wrongly revoke a perfectly correct, header-confirmed role.
            # Deliberately narrower than _is_noise_cell here: genuinely
            # wrong content (a signature name sitting in what should be a
            # time column) must still count as a real mismatch, or this
            # check could never catch that case at all.
            values = [
                r[col] for r in rows
                if col < len(r) and r[col].strip() and not _is_expected_time_placeholder(r[col].strip())
            ]
            if not values:
                continue
            match_fraction = sum(1 for v in values if _TIME_LOOSE_PATTERN.match(v)) / len(values)
            if match_fraction < self.MIN_INFERENCE_MATCH_FRACTION:
                del roles[role]

        missing = [r for r in ("date", "in_time", "out_time") if r not in roles]
        if not missing or not rows:
            return roles

        date_col = roles.get("date")
        if "in_time" in missing and date_col is not None:
            date_values = [r[date_col] for r in rows if date_col < len(r) and r[date_col].strip()]
            if date_values:
                merged_fraction = sum(
                    1 for v in date_values if _DATE_TIME_MERGED_PATTERN.match(v)
                ) / len(date_values)
                if merged_fraction >= self.MIN_INFERENCE_MATCH_FRACTION:
                    missing.remove("in_time")

        used_cols = set(roles.values())

        for role in missing:
            pattern = _VALUE_PATTERNS[role]
            for col in range(start, min(end, max_cols)):
                if col in used_cols:
                    continue
                # Excludes legitimate leave/WOFF placeholders from the
                # time-role match check the same way the validation pass
                # above does (see _is_expected_time_placeholder) -- only
                # for in_time/out_time, since applying it to "date" would
                # wrongly reject genuine dates ("/" isn't a placeholder).
                values = [
                    r[col] for r in rows
                    if col < len(r) and r[col].strip()
                    and (role == "date" or not _is_expected_time_placeholder(r[col].strip()))
                ]
                if not values:
                    continue
                # A majority, not every value, has to match -- when no
                # header row was identifiable at all (see
                # _parse_markdown_table's last resort), the real header's
                # own label text ("Date", "Sig") ends up in this same
                # probe set alongside genuine data rows, and requiring a
                # unanimous match would let that one header-shaped row
                # veto an otherwise clearly date/time-shaped column.
                match_fraction = sum(1 for v in values if pattern.match(v)) / len(values)
                if match_fraction >= self.MIN_INFERENCE_MATCH_FRACTION:
                    roles[role] = col
                    used_cols.add(col)
                    break

        return roles

    def _repair_pipe_split_date(self, cells, date_idx):
        """
        Some scans have Mistral render a date's own separator as a
        literal "|" (e.g. "01|05|26"), which collides with "|" being the
        markdown table's own cell delimiter and shreds the date into 2-3
        short numeric fragments instead of leaving it as one cell --
        silently shifting every column after it. If the cell at the
        expected date position doesn't look like a full date but merging
        it with the next 1-2 short numeric cells would, merge them back
        into one date value (and drop the now-absorbed extra cells, so
        every later column index lines up with the header again).
        """
        if date_idx is None or date_idx >= len(cells):
            return cells
        if _DATE_VALUE_PATTERN.match(cells[date_idx]):
            return cells

        for extra in (2, 1):
            fragment = cells[date_idx:date_idx + 1 + extra]
            if len(fragment) != 1 + extra:
                continue
            if not all(_NUMERIC_FRAGMENT.match(f.strip()) for f in fragment):
                continue
            merged_date = "/".join(f.strip() for f in fragment)
            if _DATE_VALUE_PATTERN.match(merged_date):
                return cells[:date_idx] + [merged_date] + cells[date_idx + 1 + extra:]

        return cells

    def _realign_row(self, cells, roles, col_range):
        """
        Individual data rows occasionally have fewer leading blank cells
        than the header implied, because Mistral's markdown generation
        sometimes drops a wholly-empty cell instead of rendering it --
        shifting every column in that one row left by however many cells
        got dropped. Most visible on wide multi-block tables, where a
        row with nothing at all in one block (see _find_group_ranges)
        can lose that block's leading blank cell entirely.

        If the expected date column's cell doesn't look like a date but
        a cell 1-3 positions earlier (still within this group's range)
        does, treat that gap as this row's own shift and apply it to
        every role -- rather than trusting the header-derived column
        positions unconditionally for every row.
        """
        date_idx = roles.get("date")
        if date_idx is None or date_idx >= len(cells):
            return roles
        if _DATE_VALUE_PATTERN.match(cells[date_idx].strip()):
            return roles

        start = col_range[0]
        for shift in (1, 2, 3):
            probe = date_idx - shift
            if probe < start or probe >= len(cells):
                continue
            if _DATE_VALUE_PATTERN.match(cells[probe].strip()):
                return {
                    role: (idx - shift if idx is not None else idx)
                    for role, idx in roles.items()
                }

        return roles

    def _rows_to_records(self, rows, roles, col_range):
        records = []
        start, end = col_range

        for row in rows:
            cells = [c.strip() for c in row.strip().strip("|").split("|")]
            cells = self._repair_pipe_split_date(cells, roles.get("date"))
            row_roles = self._realign_row(cells, roles, col_range)

            date_idx = row_roles.get("date")
            in_idx = row_roles.get("in_time")
            out_idx = row_roles.get("out_time")
            remark_idx = row_roles.get("remark")
            claimed = {i for i in (date_idx, in_idx, out_idx, remark_idx) if i is not None}

            def cell_at(idx):
                return cells[idx] if idx is not None and 0 <= idx < len(cells) else ""

            date = cell_at(date_idx)
            raw_in = cell_at(in_idx)
            raw_out = cell_at(out_idx)

            # No dedicated in_time column was found for this block (see
            # _infer_missing_roles) because the date and in-time were
            # written in the same box in the source register -- recover
            # in-time by splitting it back out of the date cell itself.
            if not raw_in:
                merged = _DATE_TIME_MERGED_PATTERN.match(date)
                if merged:
                    date, raw_in = merged.group(1), merged.group(2)

            group_cells = cells[start:end]
            if not date and not any(group_cells):
                continue

            in_time = "" if _is_noise_cell(raw_in) else raw_in
            out_time = "" if _is_noise_cell(raw_out) else raw_out

            remark = cell_at(remark_idx).strip() if remark_idx is not None else ""
            if not remark:
                # No dedicated remark column (or this row's cell in it
                # was blank) -- an overtime/extra-shift code sometimes
                # bleeds into another column instead (e.g. a signature
                # cell reading "Pinky M+B" rather than just "Pinky"),
                # recognizable by a "+"-joined code pattern regardless of
                # which column it landed in.
                for idx in range(start, min(end, len(cells))):
                    if idx in claimed:
                        continue
                    if "+" in cells[idx]:
                        remark = cells[idx].strip()
                        break

            row_data = {
                "date": date,
                "in_time": in_time,
                "out_time": out_time,
                "cells": group_cells,
            }

            # No fallback to STATUS_WORK here: StatusDetector already
            # returns "" when neither a LEAVE/WOFF keyword nor an actual
            # time was found (both in/out blank) -- AttendanceValidator's
            # own blank-row rule (in/out empty -> WOFF) is what should
            # decide that case, not a default that overrides it. An
            # earlier version of this method defaulted to STATUS_WORK
            # here, which produced "WORK" rows with no times at all
            # whenever the row had neither a recognizable time nor a
            # WOFF/LEAVE keyword Mistral's OCR happened to match.
            status = _status_detector.detect(row_data)

            # Checked only against the row's own core data (date/in/out),
            # not every cell in the block's column range -- a signature
            # column repeats the same handwritten name on every row, and
            # that handwriting sometimes has its own stroke through it
            # (a personal signature style, not a cancellation mark) that
            # Mistral renders as markdown strikethrough regardless of
            # whether that day's attendance was actually voided. Checking
            # the whole row here would flag every single row on a page
            # like that, rather than the specific day someone crossed out.
            struck_through = any(
                _STRIKETHROUGH.search(c) for c in (date, raw_in, raw_out) if c
            )

            records.append({
                "date": date,
                "in_time": in_time,
                "out_time": out_time,
                "status": status,
                "remark": remark,
                "struck_through": struck_through,
            })

        return records

    # ------------------------------------------------------------------
    # HTTP PLUMBING
    # ------------------------------------------------------------------
    # Status codes worth retrying -- rate limiting and momentary server-
    # side trouble, not anything the request itself caused (a 400/401
    # would fail identically on every retry, so isn't included here).
    _RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

    def _call_ocr(self, image_path):
        b64, ext = self._encode_image(image_path)
        payload = {
            "model": self.model,
            "document": {
                "type": "image_url",
                "image_url": f"data:image/{ext};base64,{b64}",
            },
            "include_image_base64": False,
        }

        last_error = None
        for attempt in range(1, self.OCR_MAX_ATTEMPTS + 1):
            try:
                resp = requests.post(
                    MISTRAL_OCR_URL, headers=self._headers(), json=payload, timeout=120
                )
                resp.raise_for_status()

                data = resp.json()
                pages = data.get("pages", [])
                markdown = pages[0].get("markdown", "") if pages else ""
                return self._convert_html_tables(self._unwrap_caption_json(markdown))
            except requests.exceptions.HTTPError as e:
                if e.response is None or e.response.status_code not in self._RETRYABLE_STATUS_CODES:
                    raise
                last_error = e
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                last_error = e

            if attempt < self.OCR_MAX_ATTEMPTS:
                debug_print(
                    f"MistralOCREngine: transient OCR failure on attempt "
                    f"{attempt}/{self.OCR_MAX_ATTEMPTS}, retrying: {last_error}"
                )
                time.sleep(self.OCR_RETRY_DELAY_SECONDS * attempt)

        raise last_error

    def _unwrap_caption_json(self, raw_markdown):
        """
        Mistral occasionally wraps its transcription in a bounding-box/
        caption JSON structure -- a list of {"box_2d": ..., "label": ...,
        "caption": "...markdown..."} objects -- instead of returning
        plain markdown directly for the page. Without this, the JSON
        envelope itself gets parsed as if it were the page's markdown
        text, producing garbage (a fragment of the raw JSON ending up as
        the "employee name", no usable table rows found).

        The response isn't always valid JSON to begin with (observed
        with a literal, unescaped newline inside the "caption" string,
        which plain json.loads rejects) -- so this pulls each caption's
        text out with a regex instead of requiring the wrapper to parse
        cleanly, on the theory that a markdown attendance table is
        exceedingly unlikely to itself contain the literal `"}` sequence
        that marks where a caption value ends.
        """
        stripped = raw_markdown.strip()
        if not stripped.startswith("["):
            return raw_markdown

        captions = _CAPTION_PATTERN.findall(stripped)
        if not captions:
            return raw_markdown

        return "\n\n".join(captions)

    def _convert_html_tables(self, raw_markdown):
        """
        Mistral occasionally renders one table on a page as markdown
        pipe-syntax and a *different* table on the same page as raw HTML
        (`<tr><td>...`) instead -- confirmed on a real double-page-spread
        photo where the second employee's table came back as
        `<tr><td>...</td></tr>` markup glued directly onto the end of the
        first table's last markdown row, with no newline separating them
        at all. `_TABLE_ROW` only recognizes markdown pipe rows, so
        without this, the HTML table is invisible to the parser entirely
        (silently losing a whole employee's records), and the markdown
        row it got glued onto stops matching `_TABLE_ROW` too (it no
        longer ends in "|"), losing that row as well.

        Converts each `<tr>...</tr>` sequence found anywhere in the text
        into an equivalent markdown block (header row + separator +
        data rows) in place, so it flows into the same multi-header
        parsing `_find_headers` already does for two sequential markdown
        tables. Leaves the text untouched if no `<tr>` markup is present
        at all, which is the common case.
        """
        match = re.search(r"<tr\b.*?</tr\s*>", raw_markdown, re.IGNORECASE | re.DOTALL)
        if not match:
            return raw_markdown

        end = raw_markdown.find("</table>", match.end())
        span_end = end + len("</table>") if end != -1 else match.end()
        html_block = raw_markdown[match.start():span_end]

        rows = re.findall(r"<tr\b[^>]*>(.*?)</tr\s*>", html_block, re.IGNORECASE | re.DOTALL)
        markdown_rows = []
        for row in rows:
            cells = re.findall(r"<t[dh]\b[^>]*>(.*?)</t[dh]\s*>", row, re.IGNORECASE | re.DOTALL)
            if not cells:
                continue
            cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
            markdown_rows.append("| " + " | ".join(cells) + " |")
            if len(markdown_rows) == 1:
                markdown_rows.append("| " + " | ".join("---" for _ in cells) + " |")

        if not markdown_rows:
            return raw_markdown

        converted = "\n" + "\n".join(markdown_rows) + "\n"
        return raw_markdown[:match.start()] + converted + raw_markdown[span_end:]

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _encode_image(self, image_path):
        with open(image_path, "rb") as f:
            data = f.read()
        ext = image_path.rsplit(".", 1)[-1].lower() or "jpeg"
        return base64.b64encode(data).decode("utf-8"), ext
