import base64
import json
import re
import time

import requests

from status_detector import StatusDetector
from config import debug_print

MISTRAL_OCR_URL = "https://api.mistral.ai/v1/ocr"

_status_detector = StatusDetector()

# Words that mean "this cell isn't actually a time" (a WOFF/leave marker
# sitting in the in_time/out_time field instead of a real HH:MM value).
_STATUS_NOISE = {
    "woff", "w/off", "w off", "w.off", "w/oft", "wloff",
    "leave", "leaue", "leav", "holiday", "off", "weekly off",
    "week off", "weekoff", "w|off", "wooff",
}

_PLACEHOLDER_CELL = re.compile(r"^[\s\-‐-―_.*]*$")

# A real time value, however it's formatted, is made up of digits plus
# separators/am-pm markers -- nothing else. Stripping all of that out and
# finding leftover letters means the field actually holds a garbled status
# word (e.g. "W-0661") rather than a time, even when it doesn't match any
# known misspelling in _STATUS_NOISE. Includes "'" -- a handwritten colon
# sometimes gets OCR'd as an apostrophe right next to the real separator
# ("6'.50" for "6:50"), which would otherwise look like leftover non-time
# text on an actual time value.
_TIME_RESIDUE = re.compile(r"\d+|[:.,\s\-']|a\.?m\.?|p\.?m\.?", re.IGNORECASE)


def _has_non_time_residue(text):
    return bool(_TIME_RESIDUE.sub("", text).strip())


def _is_status_noise(text):
    """True if a field's text is a status word, not a time."""
    t = text.lower().strip().replace("/", " ").replace(".", " ")
    return any(noise in t for noise in _STATUS_NOISE)


def _is_expected_time_placeholder(text):
    """
    True if a blank in_time/out_time field is *expected* to be blank -- a
    recognizable status word (WOFF/leave/off) or a bare placeholder mark
    (blank, dash).
    """
    return _is_status_noise(text) or bool(_PLACEHOLDER_CELL.match(text))


def _is_noise_value(text):
    """
    True if an in_time/out_time value isn't actually a time -- either a
    recognizable status word/placeholder, or (more generally) any text
    that has leftover letters once every digit/separator/am-pm token is
    stripped out, which is what a status word garbled during OCR looks
    like even when it doesn't match a known misspelling.
    """
    return _is_expected_time_placeholder(text) or _has_non_time_residue(text)


# A remark only ever means "this employee worked more than one shift the
# same day", written as each shift's single-letter code joined by "+"
# (e.g. "A+B" for two shifts, "A+B+C" for three) -- never free text. This
# is deliberately a \b-bounded single-letter run, not just "contains a
# +": a remark field sometimes has other text merged in with it (e.g.
# "Pinky M+B", where only "M+B" is the real remark and "Pinky" is a
# signature that bled into the same field), and some other non-remark
# text coincidentally containing a "+" (e.g. "M+Bag") isn't this shape at
# all -- "Bag" is a whole word, not a single shift-code letter, so the
# trailing \b never lands right after a lone letter and the match fails
# outright.
_SHIFT_CODE_REMARK = re.compile(r"\b([A-Za-z](?:\+[A-Za-z])+)\b")

# A real employee name never contains a run of digits -- a phone number,
# employee code, or serial number occasionally bleeding into the model's
# own name answer is noise, not part of the name. Floor of 3 so a single
# stray digit doesn't nuke an otherwise-fine name.
_DIGIT_RUN = re.compile(r"\d{3,}")

# Left behind once a parenthesized serial number's digits are stripped
# out (e.g. "(77)" -> "()") -- never meaningful on its own.
_EMPTY_PARENS = re.compile(r"\(\s*\)")

# A "(Sub)" designation tag -- marking a substitute/temporary worker --
# sometimes rides along with the real name (e.g. "(SUB) PINKU CHOUDHARY")
# and isn't part of the name itself.
_NOISE_TAG = re.compile(r"\(\s*sub\s*\)", re.IGNORECASE)

# A trade/grade code ("MST", "M.S.T", "MS") sometimes rides along with
# the name, with its own small serial number right in front of it (e.g.
# "(7) MST", "Deepak Choubray 1 MS", "M.S.T Pavi Kumar").
_TRADE_CODE_NOISE = re.compile(r"\(?\s*\d{0,2}\s*\)?\s*m\.?\s*s\.?\s*t?\.?\b", re.IGNORECASE)


def _clean_employee_name(text):
    """
    Light safety-net cleanup on whatever name the model returns -- it's
    already been asked to read specifically from the table's own heading/
    signature text, so this isn't hunting through ambiguous candidates
    the way the old markdown-heuristic version had to, just mopping up
    noise (a stray code/tag/phone digits) that occasionally rides along
    with an otherwise-correct answer.
    """
    if not text:
        return ""
    cleaned = _DIGIT_RUN.sub("", text)
    cleaned = _NOISE_TAG.sub("", cleaned)
    cleaned = _TRADE_CODE_NOISE.sub("", cleaned)
    cleaned = _EMPTY_PARENS.sub("", cleaned)
    # Whatever's left in parens is real name text a ruled box or stray
    # mark caused the model to wrap in "(" ")" -- drop just the
    # punctuation, not the letters inside it.
    cleaned = cleaned.replace("(", "").replace(")", "")
    cleaned = re.sub(r"^[\s:./\-]+|[\s:./\-]+$", "", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


# Confirmed with the client: an employee code is always 4-6 digits,
# written anywhere in the header (not a fixed position/box), and a
# ceiling of 6 rules out a 10-digit phone number or a longer reference
# number without needing to recognize either as such. Validated rather
# than trusted outright since the model can still occasionally answer
# with a phone number or other stray digits despite being told what shape
# a real code has.
_EMP_CODE_SHAPE = re.compile(r"^\d{4,6}$")


def _clean_employee_code(text):
    text = (text or "").strip()
    digits = re.sub(r"\D", "", text)
    return digits if _EMP_CODE_SHAPE.match(digits) else ""


# The schema every extraction call requests -- one entry per employee on
# the page (almost always one, occasionally two sharing one wide table),
# each with its own name/code and full list of daily records. Asking the
# model for structured fields directly (rather than a markdown table this
# class then has to parse itself) sidesteps a whole class of bugs the
# markdown-parsing approach had: shifted/duplicated column headers,
# ambiguous in/out direction when a second table's header mislabels its
# own columns, "|" characters inside a cell shredding a date across
# phantom columns, etc. -- confirmed on a real page with two employees
# sharing one wide table (whose header had a missing serial-number cell
# for the second employee, shifting every later label one column off)
# that the old markdown parser lost the second employee's Out Time
# entirely and got the first employee's In/Out direction backwards;
# structured extraction with an explicit prompt describing the dual-table
# layout got every row right, including direction, on the same page.
_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "employees": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "employee_name": {"type": "string"},
                    "employee_code": {"type": "string"},
                    "records": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "serial_number": {"type": "string"},
                                "date": {"type": "string"},
                                "in_time": {"type": "string"},
                                "out_time": {"type": "string"},
                                "remark": {"type": "string"},
                                "struck_through": {"type": "boolean"},
                            },
                            "required": [
                                "serial_number", "date", "in_time",
                                "out_time", "remark", "struck_through",
                            ],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["employee_name", "employee_code", "records"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["employees"],
    "additionalProperties": False,
}

_EXTRACTION_PROMPT = (
    "Extract every employee attendance table on this page. Most pages "
    "have exactly one employee; some pages hold two separate employees "
    "sharing one wide row of columns side by side (a left table and a "
    "right table sharing one header row), or two full tables one after "
    "another -- if so, extract each employee as its own separate entry "
    "in the employees array, never merged into one. Include every row "
    "even if it's incomplete (a blank day, a leave/week-off day, or a "
    "day with no time filled in at all) -- represent any missing field "
    "as an empty string rather than omitting the row. Read employee_name "
    "from the clearly written heading directly above that specific "
    "employee's own table -- the name written once, larger, near the top "
    "of that employee's section (often next to a phone number) -- not "
    "from a page title that might belong to the other employee if the "
    "page holds two. Do NOT read employee_name from the table's own "
    "repeated \"Name\"/signature column, where the same person re-signs "
    "on every single row -- that's written quickly and inconsistently "
    "row to row and is far less reliable than the one clear heading. "
    "Only use that signature column as employee_name if there is truly "
    "no heading name anywhere above the table. employee_code is a bare "
    "4-6 digit number, handwritten somewhere in that employee's own "
    "header area -- it isn't always in a fixed spot next to the name, so "
    "check the whole header region for it. It is never a phone number "
    "(usually 10 digits, and never the employee_name heading's own "
    "number) and never the number in the ruled \"Page No.\" / \"Date\" "
    "box in the corner of the header -- that box (along with its printed "
    "label) is part of the notebook's own pre-printed page template, not "
    "something written about the employee, and must never be used as "
    "employee_code even when it has a number in it. If no genuine 4-6 "
    "digit handwritten code is visible anywhere in the header, leave "
    "employee_code as an empty string rather than guessing. "
    "struck_through is true only for a "
    "row whose date/time cells are visibly crossed out/voided in the "
    "handwriting itself, not a person's own signature style."
)


class MistralOCREngine:
    """
    Full-page OCR engine for the pipeline: Mistral's hosted Document AI
    OCR endpoint provides the page read (records + employee name/code) as
    structured JSON matching _EXTRACTION_SCHEMA, via `document_annotation_
    format` -- not a markdown transcription this class then has to parse
    itself. Surya/EasyOCR were removed from the pipeline entirely (see
    git history on llama-vision-integration/main for that version) once
    benchmarking showed Mistral's date/time reads were reliable enough to
    stand alone as the primary engine; the markdown-table-parsing
    approach that followed was itself replaced by this structured
    extraction (see _EXTRACTION_SCHEMA's docstring for why).

    Talks to Mistral's REST API directly with `requests` rather than the
    `mistralai` SDK: the SDK pins httpx>=0.28.1, which conflicted with
    surya-ocr's httpx<0.28 requirement when both lived in the same venv
    (confirmed while building the standalone test in mistral_ocr_test/).
    Now that Surya is gone this conflict can't recur, but there's no
    reason to add the SDK's dependency footprint back for a single REST
    call this class already makes directly.

    Does NOT recheck its own flagged rows -- see vision_recheck_engine.py
    for why that's a genuinely different model (Qwen 3.6 27B via Groq)
    instead of Mistral re-reading itself, and recheck_utils.py for the
    engine-agnostic outlier/format gate deciding which rows get
    rechecked at all.
    """

    # See run(): a single employee with more records than this is treated
    # as a corrupted response (padding/repetition) rather than genuine
    # data -- a month's register realistically tops out around 31-35 rows
    # even with a few duplicate-date extra shifts.
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

    # ------------------------------------------------------------------
    # FULL-PAGE OCR
    # ------------------------------------------------------------------
    def run(self, image_path):
        """
        Returns a list of "blocks" -- almost always exactly one, but more
        than one when a page holds multiple employees. Each block is
        {"records": [...], "signature_name": "...", "employee_code": "..."}
        -- records is a list of record dicts (date, in_time, out_time,
        status, remark, struck_through).

        Returns [] if the call fails or no usable data is found at all.
        """
        try:
            employees = self._call_ocr(image_path)

            blocks = [self._build_block(emp) for emp in employees]
            blocks = [b for b in blocks if b["records"]]

            # A real month's attendance register has at most a few dozen
            # rows -- confirmed on a real page that the model can pad a
            # mostly-correct response with extra blank/placeholder rows
            # past the real data. Rather than silently returning a block
            # flooded with junk records, treat an implausibly large block
            # as evidence the whole response was corrupted and let the
            # caller's existing empty-result handling (retry with a
            # different image) take over instead.
            if any(len(b["records"]) > self.MAX_PLAUSIBLE_RECORDS_PER_BLOCK for b in blocks):
                debug_print(
                    "MistralOCREngine: implausibly large block "
                    "(likely a corrupted/padded response), discarding"
                )
                return []

            return blocks
        except Exception as e:
            debug_print(f"MistralOCREngine: full-page call failed: {e}")
            return []

    def _build_block(self, employee):
        raw_records = employee.get("records", []) or []
        records = []

        for raw in raw_records:
            date = (raw.get("date") or "").strip()
            in_time = (raw.get("in_time") or "").strip()
            out_time = (raw.get("out_time") or "").strip()
            remark_raw = (raw.get("remark") or "").strip()

            # A padding row past the real data (see run()'s size guard) --
            # nothing to record here at all.
            if not date and not in_time and not out_time and not remark_raw:
                continue

            in_time = "" if _is_noise_value(in_time) else in_time
            out_time = "" if _is_noise_value(out_time) else out_time

            remark_match = _SHIFT_CODE_REMARK.search(remark_raw) if remark_raw else None
            remark = remark_match.group(1) if remark_match else ""

            row_data = {
                "date": date,
                "in_time": in_time,
                "out_time": out_time,
                "cells": [remark_raw],
            }

            # No fallback to STATUS_WORK here: StatusDetector already
            # returns "" when neither a LEAVE/WOFF keyword nor an actual
            # time was found (both in/out blank) -- AttendanceValidator's
            # own blank-row rule is what should decide that case, not a
            # default that overrides it.
            status = _status_detector.detect(row_data)

            records.append({
                "date": date,
                "in_time": in_time,
                "out_time": out_time,
                "status": status,
                "remark": remark,
                "struck_through": bool(raw.get("struck_through")),
            })

        return {
            "records": records,
            "signature_name": _clean_employee_name(employee.get("employee_name", "")),
            "employee_code": _clean_employee_code(employee.get("employee_code", "")),
        }

    # ------------------------------------------------------------------
    # HTTP PLUMBING
    # ------------------------------------------------------------------
    # Status codes worth retrying -- rate limiting and momentary server-
    # side trouble, not anything the request itself caused (a 400/401
    # would fail identically on every retry, so isn't included here).
    _RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

    def _call_ocr(self, image_path):
        """
        Returns the "employees" list from the structured document
        annotation -- see _EXTRACTION_SCHEMA. Raises on failure (caught
        by run()).
        """
        b64, ext = self._encode_image(image_path)
        payload = {
            "model": self.model,
            "document": {
                "type": "image_url",
                "image_url": f"data:image/{ext};base64,{b64}",
            },
            "document_annotation_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "attendance_extraction",
                    "schema": _EXTRACTION_SCHEMA,
                    "strict": True,
                },
            },
            "document_annotation_prompt": _EXTRACTION_PROMPT,
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
                annotation = json.loads(data.get("document_annotation") or "{}")
                return annotation.get("employees", [])
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
