import re
import statistics
from datetime import datetime, timedelta
from dateutil import parser as dateutil_parser

from constants import STATUS_WORK, OFF_STATUSES


class ValidationEngine:
    """
    Modular post-extraction validation layer.
    Flags suspicious cells without modifying OCR output.
    Does not alter pipeline architecture or OCR engine.

    A highlighted cell is not a claim that the value is wrong -- it's a
    claim that the cell is worth a human looking at. HR reviews only the
    highlighted cells, so the question every check answers is "would I
    ask a human to look at this?", not "can I prove this is incorrect?".

    Three-stage design:
      1. Checks write *evidence* into record["validation"] (OCR-mistake
         signals) or record["attendance_analytics"] (behavioral/statistical
         signals) -- they never decide highlighting directly.
      2. _decide_highlights() is the single place that turns evidence into
         record["cell_highlight"].
      3. attendance_analytics never influences highlighting. It exists for
         future behavioral reporting (late coming, overtime, shift
         changes) and is kept structurally separate from OCR validation on
         purpose -- "this shift is unusual" and "this cell is probably a
         misread" are different questions, and conflating them is what
         caused legitimate shift changes to get flagged as OCR errors.
    """

    # Statuses that represent a day with no attendance times at all.
    OFF_STATUSES = OFF_STATUSES

    # yyyy/mm/dd -- matches post_processing.normalize_date's output shape.
    DATE_PATTERN = re.compile(r"^(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})$")

    # Date-sequence check (see _flag_date_sequence_issues): how many days
    # a forward gap between two dated rows is allowed to span before it
    # stops looking like "a few unwritten leave/WOFF rows" and starts
    # looking like a misread date.
    MAX_FORWARD_GAP_DAYS = 10

    # An out_time earlier than in_time is only plausible for a shift that
    # crosses midnight; beyond this many hours it's treated as a real error.
    MAX_OVERNIGHT_HOURS = 16

    # Duration-outlier check (see _flag_duration_outliers): need enough
    # WORK rows on the sheet for "typical shift length" to mean anything.
    MIN_SAMPLES_FOR_DURATION_STATS = 5

    # Robust modified z-score cutoff (Iglewicz & Hoaglin). ~3.5 is the
    # standard threshold for "this point doesn't belong to the cluster".
    DURATION_MAD_THRESHOLD = 3.5

    # Same cutoff, applied to raw in/out clock times instead of duration.
    TIME_MAD_THRESHOLD = 3.5

    # Shift-pattern grouping for the time-of-day check: in_time values are
    # chained into the same group unless a gap this large (in minutes)
    # separates them, so a person with several real shift patterns on the
    # same sheet (night/day/evening) is compared against their own pattern
    # instead of the whole sheet at once.
    SHIFT_CLUSTER_GAP_MINUTES = 240

    # Same reasoning as MIN_SAMPLES_FOR_DURATION_STATS, applied per cluster.
    MIN_SAMPLES_FOR_TIME_STATS = 5

    # Shift-pattern override for chronology_invalid (see
    # _flag_shift_pattern_overrides): how many consecutive WORK rows with
    # matching clock times are needed before a long overnight gap is
    # treated as a real (if unmarked on the page) shift change rather
    # than a one-off misread.
    MIN_SHIFT_PATTERN_RUN = 3
    SHIFT_PATTERN_TOLERANCE_MINUTES = 90

    # ------------------------------------------------------------------
    # PER-CELL CONFIDENCE SCORE
    #
    # cell_highlight (above) answers "does a human need to look at this
    # cell" with a yes/no. This score exists alongside it, not instead of
    # it, to answer "how much should the human trust it" -- a 0-100
    # number an operator can sort/triage by, built from exactly the same
    # evidence (validation flags, cross-engine agreement) rather than any
    # new signal. needs_review in the output still comes from
    # cell_highlight; the score is explanatory, not a second decision-maker.
    # ------------------------------------------------------------------
    FIELD_CONFIDENCE_BASE = 100
    AGREEMENT_BONUS = 40       # Mistral and the vision recheck model independently agree
    VALID_FORMAT_BONUS = 20
    OCR_DISAGREEMENT_PENALTY = 40
    INVALID_FORMAT_PENALTY = 30
    MISSING_PENALTY = 50

    # ------------------------------------------------------------------
    # PER-ROW CONFIDENCE SCORE
    #
    # A single 0-100 number per record, for sorting/triaging whole rows
    # rather than individual cells -- built the same way field_confidence
    # is: start at 100, subtract for whichever row-wide issues actually
    # fired. Deliberately excludes the six field-specific validation
    # flags (in_time/out_time invalid, missing, or disagreement) since
    # those already lower field_confidence, and this score folds those
    # in separately by taking the lowest of the two field scores --
    # penalizing them here too would count the same evidence twice.
    # ------------------------------------------------------------------
    ROW_CONFIDENCE_PENALTIES = {
        "date_invalid": 50,
        "chronology_invalid": 30,
        "time_alignment_suspect": 20,
        "status_time_mismatch": 20,
        "date_sequence_invalid": 20,
        "struck_through": 100,
    }

    # ------------------------------------------------------------------
    # SEVERITY MODEL (OCR validation only -- attendance_analytics is a
    # separate namespace and never appears here)
    #
    # CRITICAL: objectively wrong regardless of context -- date_invalid,
    # impossible chronology, a mandatory time missing, etc. These are
    # certainties, not judgment calls.
    #
    # STRONG: the independent vision recheck model (see
    # vision_recheck_engine.py) disagreeing with Mistral's first read of
    # the same cell. This is NOT
    # a claim the first read was wrong -- re-reading a small/ambiguous
    # region can go either way. It highlights anyway, because the bar
    # for "worth a human glance" is lower than the bar for "provably
    # incorrect": two reads of the same cell landing on different values
    # is reason enough to ask a person to look, even when neither read is
    # provably wrong on its own. Both tiers render with the same single
    # highlight color -- the distinction is internal bookkeeping for
    # why a cell was flagged, not a UI signal.
    # ------------------------------------------------------------------
    CRITICAL_EVIDENCE = {
        "date_invalid": ["date"],
        "in_time_invalid": ["in_time"],
        "out_time_invalid": ["out_time"],
        "in_time_missing": ["in_time"],
        "out_time_missing": ["out_time"],
        "time_alignment_suspect": ["in_time", "status"],
        "status_time_mismatch": ["status"],
        "date_sequence_invalid": ["date"],
        # A row the model itself reported as struck_through (see
        # MistralOCREngine._EXTRACTION_SCHEMA) looks like it was voided/
        # corrected by the person who wrote it -- worth a human glance on
        # the whole row, not just one field. Only catches what the model
        # itself flagged; a crossed-out row it doesn't notice as such
        # won't trigger this.
        "struck_through": ["date", "in_time", "out_time", "status"],
    }

    STRONG_EVIDENCE = {
        "in_time_ocr_disagreement": ["in_time"],
        "out_time_ocr_disagreement": ["out_time"],
    }

    def validate_records(self, records):
        validated = [self._validate_single(record) for record in records]

        self._flag_date_sequence_issues(validated)
        self._flag_duration_outliers(validated)
        self._flag_time_of_day_outliers(validated)
        self._flag_shift_pattern_overrides(validated)

        for record in validated:
            self._decide_highlights(record)
            self._compute_field_confidence(record)
            self._compute_row_confidence(record)

        for record in validated:
            self._compact_for_output(record)

        return validated

    # ------------------------------------------------------------------
    # PER-RECORD VALIDATION (OCR evidence only)
    # ------------------------------------------------------------------
    def _validate_single(self, record):
        flags = {
            "date_invalid": False,
            "in_time_invalid": False,
            "out_time_invalid": False,
            "in_time_missing": False,
            "out_time_missing": False,
            "chronology_invalid": False,
            "time_alignment_suspect": False,
            "status_time_mismatch": False,
            "date_sequence_invalid": False,
            "in_time_ocr_disagreement": bool(record.get("in_time_ocr_disagreement")),
            "out_time_ocr_disagreement": bool(record.get("out_time_ocr_disagreement")),
            "struck_through": bool(record.get("struck_through")),
        }

        date_str = record.get("date", "")
        in_time_str = record.get("in_time", "")
        out_time_str = record.get("out_time", "")
        status = record.get("status", "")

        # ---------- DATE VALIDATION ----------
        if date_str and self._parse_date_obj(date_str) is None:
            flags["date_invalid"] = True

        # ---------- TIME VALIDATION ----------
        in_time_parsed = self._parse_time(in_time_str) if in_time_str else None
        out_time_parsed = self._parse_time(out_time_str) if out_time_str else None

        if in_time_str and in_time_parsed is None:
            flags["in_time_invalid"] = True

        if out_time_str and out_time_parsed is None:
            flags["out_time_invalid"] = True

        # ---------- MISSING VALUES ----------
        if status == STATUS_WORK:
            if not in_time_str:
                flags["in_time_missing"] = True
            if not out_time_str:
                flags["out_time_missing"] = True

        # ---------- CHRONOLOGY VALIDATION ----------
        if in_time_parsed is not None and out_time_parsed is not None:
            if not self._is_chronological(in_time_parsed, out_time_parsed):
                flags["chronology_invalid"] = True

        # ---------- TIME ALIGNMENT SUSPECT ----------
        if not in_time_str and out_time_str and status not in self.OFF_STATUSES:
            flags["time_alignment_suspect"] = True

        # ---------- STATUS / TIME MISMATCH ----------
        has_any_time = bool(in_time_str or out_time_str)
        if (status in self.OFF_STATUSES and has_any_time) or (not status and has_any_time):
            flags["status_time_mismatch"] = True

        record["validation"] = flags
        record["attendance_analytics"] = {
            "duration_outlier": False,
            "in_time_outlier": False,
            "out_time_outlier": False,
        }

        # Their only purpose was feeding the copy above -- pipeline.py
        # sets these on the record purely as a hand-off to this method,
        # not as data meant for the final output. Once compacted,
        # "in_time_ocr_disagreement" appearing in the validation list
        # already carries this same signal, so keeping the top-level
        # copy too would just be paying twice for one fact.
        record.pop("in_time_ocr_disagreement", None)
        record.pop("out_time_ocr_disagreement", None)

        return record

    # ------------------------------------------------------------------
    # DATE SEQUENCE VALIDATION
    # ------------------------------------------------------------------
    def _flag_date_sequence_issues(self, records):
        """
        Register rows are written one calendar day at a time, so dates
        should never go backwards and shouldn't jump implausibly far
        forward -- but two real-world patterns mean it's *not* true that
        every row must be exactly one day after the last:

        - An employee working an extra/overnight shift gets a second row
          for the *same* date, not the next one.
        - On a leave/WOFF day, some registers leave the date column
          blank entirely rather than writing it, so a run of several
          unwritten days is a real, ordinary gap, not a misread.

        So this only flags a date that goes backwards, or forward by an
        implausible amount (MAX_FORWARD_GAP_DAYS) -- a same-day repeat or
        a modest forward gap is left alone. Compare against the
        immediately preceding *dated* row (not just the last "good" one)
        so a genuine backwards jump gets caught at every bad step.
        """
        previous_date = None

        for record in records:

            parsed = self._parse_date_obj(record.get("date", ""))

            if parsed is None:
                continue

            if previous_date is not None:
                gap_days = (parsed - previous_date).days
                if gap_days < 0 or gap_days > self.MAX_FORWARD_GAP_DAYS:
                    record["validation"]["date_sequence_invalid"] = True

            previous_date = parsed

    # ------------------------------------------------------------------
    # ATTENDANCE ANALYTICS: DURATION OUTLIER
    #
    # NOT OCR validation. A handwritten hour digit misread as a different-
    # but-plausible hour (e.g. "18:26" read as "19:26") produces exactly
    # the same signal as a genuine shift change -- there is no way to
    # tell those apart from the extracted text alone. This check answers
    # "is this shift unusual", not "is this cell likely a misread", so it
    # never contributes to cell_highlight (see class docstring).
    # ------------------------------------------------------------------
    def _flag_duration_outliers(self, records):
        durations = []

        for record in records:
            if record.get("status") != STATUS_WORK:
                continue

            flags = record.get("validation", {})
            if flags.get("chronology_invalid") or flags.get("in_time_invalid") or flags.get("out_time_invalid"):
                continue  # already-flagged/unparseable rows would pollute the stats

            duration = self._record_duration_minutes(record)
            if duration is not None:
                durations.append((record, duration))

        if len(durations) < self.MIN_SAMPLES_FOR_DURATION_STATS:
            return

        values = [d for _, d in durations]
        median = statistics.median(values)
        mad = statistics.median(abs(v - median) for v in values)

        if mad == 0:
            return  # every shift is identical length -- no spread to compare against

        for record, duration in durations:
            modified_z = 0.6745 * (duration - median) / mad
            if abs(modified_z) >= self.DURATION_MAD_THRESHOLD:
                record["attendance_analytics"]["duration_outlier"] = True

    # ------------------------------------------------------------------
    # ATTENDANCE ANALYTICS: TIME-OF-DAY OUTLIER
    #
    # Same reasoning as _flag_duration_outliers -- an unusual clock-in
    # time is behavioral information (or, at best, a coincidental
    # byproduct of a misread), not direct OCR evidence. Grouped into
    # shift clusters by in_time (single-linkage, >=4h gap starts a new
    # cluster) so a person with several real shift patterns on the same
    # sheet (night/day/evening) is compared against their own pattern
    # instead of the whole sheet at once.
    # ------------------------------------------------------------------
    def _flag_time_of_day_outliers(self, records):
        candidates = []

        for record in records:
            if record.get("status") != STATUS_WORK:
                continue

            flags = record.get("validation", {})
            if flags.get("in_time_invalid") or flags.get("in_time_missing"):
                continue

            in_parsed = self._parse_time(record.get("in_time", ""))
            if in_parsed is None:
                continue

            candidates.append((record, in_parsed[0] * 60 + in_parsed[1]))

        if len(candidates) < self.MIN_SAMPLES_FOR_TIME_STATS:
            return

        candidates.sort(key=lambda pair: pair[1])

        clusters = [[candidates[0]]]
        for pair in candidates[1:]:
            if pair[1] - clusters[-1][-1][1] > self.SHIFT_CLUSTER_GAP_MINUTES:
                clusters.append([])
            clusters[-1].append(pair)

        for cluster in clusters:
            if len(cluster) < self.MIN_SAMPLES_FOR_TIME_STATS:
                continue

            self._flag_column_outliers(cluster, "in_time_outlier", "in_time")
            self._flag_column_outliers(cluster, "out_time_outlier", "out_time")

    def _flag_column_outliers(self, cluster, flag_name, field):
        values = []

        for record, _ in cluster:
            flags = record.get("validation", {})
            if flags.get(f"{field}_invalid") or flags.get(f"{field}_missing"):
                continue

            parsed = self._parse_time(record.get(field, ""))
            if parsed is None:
                continue

            values.append((record, parsed[0] * 60 + parsed[1]))

        if len(values) < self.MIN_SAMPLES_FOR_TIME_STATS:
            return

        minutes = [m for _, m in values]
        median = statistics.median(minutes)
        mad = statistics.median(abs(m - median) for m in minutes)

        if mad == 0:
            return

        for record, minute in values:
            modified_z = 0.6745 * (minute - median) / mad
            if abs(modified_z) >= self.TIME_MAD_THRESHOLD:
                record["attendance_analytics"][flag_name] = True

    # ------------------------------------------------------------------
    # SHIFT-PATTERN OVERRIDE FOR CHRONOLOGY_INVALID
    #
    # _is_chronological already tolerates an overnight shift up to
    # MAX_OVERNIGHT_HOURS -- but some registers have no AM/PM marker at
    # all, and a genuine night shift (in ~22:00, out ~07:00) reads as a
    # ~21-hour gap literally, past that cap. A single row like that is
    # still most likely a misread; several *consecutive* WORK rows with
    # matching clock times sharing the same "impossible" gap is much more
    # likely a real (if unmarked) shift change than N independent errors
    # that all happen to look identical. Only rows inside such a run get
    # un-flagged -- an isolated one-off long gap stays flagged.
    # ------------------------------------------------------------------
    def _flag_shift_pattern_overrides(self, records):
        run = []

        def flush_run():
            if len(run) >= self.MIN_SHIFT_PATTERN_RUN:
                for rec in run:
                    rec["validation"]["chronology_invalid"] = False
            run.clear()

        prev_in_minutes = prev_out_minutes = None

        for record in records:
            # A day off in the middle of a run (e.g. a WOFF between two
            # night-shift days) doesn't break the pattern -- skip it
            # without resetting the run or its reference clock times.
            if record.get("status") in self.OFF_STATUSES:
                continue

            flags = record.get("validation", {})

            if record.get("status") != STATUS_WORK or not flags.get("chronology_invalid"):
                flush_run()
                prev_in_minutes = prev_out_minutes = None
                continue

            in_parsed = self._parse_time(record.get("in_time", ""))
            out_parsed = self._parse_time(record.get("out_time", ""))
            if in_parsed is None or out_parsed is None:
                flush_run()
                prev_in_minutes = prev_out_minutes = None
                continue

            in_minutes = in_parsed[0] * 60 + in_parsed[1]
            out_minutes = out_parsed[0] * 60 + out_parsed[1]

            matches_run = (
                prev_in_minutes is not None
                and abs(in_minutes - prev_in_minutes) <= self.SHIFT_PATTERN_TOLERANCE_MINUTES
                and abs(out_minutes - prev_out_minutes) <= self.SHIFT_PATTERN_TOLERANCE_MINUTES
            )

            if not matches_run:
                flush_run()

            run.append(record)
            prev_in_minutes, prev_out_minutes = in_minutes, out_minutes

        flush_run()

    def _record_duration_minutes(self, record):
        """Shift length in minutes, handling overnight shifts. None if unparseable."""
        in_parsed = self._parse_time(record.get("in_time", ""))
        out_parsed = self._parse_time(record.get("out_time", ""))

        if in_parsed is None or out_parsed is None:
            return None

        in_minutes = in_parsed[0] * 60 + in_parsed[1]
        out_minutes = out_parsed[0] * 60 + out_parsed[1]

        if out_minutes >= in_minutes:
            return out_minutes - in_minutes
        return (1440 - in_minutes) + out_minutes

    # ------------------------------------------------------------------
    # DECISION LAYER: evidence -> cell_highlight
    #
    # The single place severity rules are applied. Everything above this
    # only ever writes to record["validation"] or
    # record["attendance_analytics"]; nothing above decides highlighting.
    #
    # cell_highlight = CRITICAL evidence OR STRONG evidence.
    # attendance_analytics is deliberately never consulted here -- see
    # the SEVERITY MODEL comment above for why.
    # ------------------------------------------------------------------
    def _decide_highlights(self, record):
        validation = record["validation"]
        hit_fields = set()

        for flag_name, fields in self.CRITICAL_EVIDENCE.items():
            if validation.get(flag_name):
                hit_fields.update(fields)

        for flag_name, fields in self.STRONG_EVIDENCE.items():
            if validation.get(flag_name):
                hit_fields.update(fields)

        if validation.get("chronology_invalid"):
            out_time_parsed = self._parse_time(record.get("out_time", ""))
            chrono_ambiguous = out_time_parsed is not None and out_time_parsed[0] < 12
            hit_fields.add("out_time")
            if not chrono_ambiguous:
                hit_fields.add("in_time")

        # A list of only the flagged cells, not a 4-key dict spelling out
        # "false" for every cell that *isn't* flagged -- the vast
        # majority of rows flag nothing at all, so the all-false shape
        # was pure overhead. "row_highlight" (== has_issue) and
        # "row_fully_invalid" (== every cell in this list) are dropped
        # entirely as pure duplicates; a caller derives the latter with
        # len(cell_highlight) == 4 if it needs it (see excel_report.py).
        record["cell_highlight"] = [
            field for field in ("date", "in_time", "out_time", "status")
            if field in hit_fields
        ]

        record["has_issue"] = bool(record["cell_highlight"])

    # ------------------------------------------------------------------
    # PER-CELL CONFIDENCE SCORE (see FIELD_CONFIDENCE_BASE etc. above)
    # ------------------------------------------------------------------
    def _compute_field_confidence(self, record):
        status = record.get("status", "")
        validation = record.get("validation", {})
        highlight = record.get("cell_highlight", [])

        confidence = {}

        for field in ("in_time", "out_time"):

            if status in self.OFF_STATUSES:
                # No time is *expected* on a Leave/WOFF row -- an empty
                # field here is correct, not a defect. Routine case,
                # nothing to report -- omitted below like any other
                # default-confidence field.
                continue

            score = self.FIELD_CONFIDENCE_BASE
            reasons = []

            if validation.get(f"{field}_missing"):
                score -= self.MISSING_PENALTY
                reasons.append("empty")
            else:
                if validation.get(f"{field}_invalid"):
                    score -= self.INVALID_FORMAT_PENALTY
                    reasons.append("invalid format")
                else:
                    score += self.VALID_FORMAT_BONUS
                    reasons.append("valid format")

                if validation.get(f"{field}_ocr_disagreement"):
                    score -= self.OCR_DISAGREEMENT_PENALTY
                    reasons.append("recheck disagreement")
                elif record.get(f"{field}_recheck_value"):
                    # This cell was actually rechecked (not every cell is
                    # -- see recheck_utils.needs_recheck) and Mistral's
                    # original read agreed with the independent recheck.
                    score += self.AGREEMENT_BONUS
                    reasons.append("Mistral/recheck read agree")

            # A perfect score with nothing but the routine "valid
            # format" reason is the common case (most cells are never
            # even rechecked) and carries no signal beyond what
            # cell_highlight already says -- omitted to avoid paying
            # bytes for "everything's fine" on the majority of cells.
            # Anything else (a penalty, or a recheck that actually ran)
            # is kept in full, since that's exactly what a reviewer
            # triaging flagged cells needs to see.
            if reasons == ["valid format"]:
                continue

            confidence[field] = {
                "score": max(0, min(100, score)),
                "needs_review": field in highlight,
                "reasons": reasons,
            }

        record["field_confidence"] = confidence

    # ------------------------------------------------------------------
    # PER-ROW CONFIDENCE SCORE (see ROW_CONFIDENCE_PENALTIES above)
    # ------------------------------------------------------------------
    def _compute_row_confidence(self, record):
        validation = record["validation"]  # still a dict at this point
        score = self.FIELD_CONFIDENCE_BASE

        for issue, penalty in self.ROW_CONFIDENCE_PENALTIES.items():
            if validation.get(issue):
                score -= penalty

        # Folds in in_time/out_time's own scores rather than re-deriving
        # them -- a row is only as trustworthy as its least trustworthy
        # cell. A field missing from field_confidence (the routine case,
        # see _compute_field_confidence) is implicitly a perfect 100 and
        # doesn't need to be looked up.
        for field in ("in_time", "out_time"):
            field_confidence = record["field_confidence"].get(field)
            if field_confidence is not None:
                score = min(score, field_confidence["score"])

        record["row_confidence"] = max(0, min(100, score))

    # ------------------------------------------------------------------
    # FINAL COMPACTION -- validation/attendance_analytics are dicts with
    # every possible key always present (mostly "false") throughout all
    # the processing above, since the internal checks rely on dict
    # semantics (.get(), direct key assignment). Only at the very end,
    # once nothing else needs to read them as dicts, are they rewritten
    # into lists of just the checks that actually fired -- confirmed on
    # a real page that 86% of rows flag nothing at all, so the all-keys
    # shape was spending bytes on "false" that carries no signal, not on
    # real evidence. A genuinely flagged row keeps every fired check
    # named here; nothing about which cells get highlighted changes.
    #
    # There used to be a "verification" field here too (a per-field
    # {"mistral": ..., "recheck": ...} view) -- removed as pure
    # duplication, since it never held anything not already on the
    # record directly (the field's own value, and *_recheck_value when
    # present).
    # ------------------------------------------------------------------
    def _compact_for_output(self, record):
        record["validation"] = [k for k, v in record["validation"].items() if v]
        record["attendance_analytics"] = [k for k, v in record["attendance_analytics"].items() if v]

    # ------------------------------------------------------------------
    # DATE PARSING
    # ------------------------------------------------------------------
    def _parse_date_obj(self, date_str):
        date_str = (date_str or "").strip()

        match = self.DATE_PATTERN.match(date_str)
        if not match:
            return None

        day, month, year = match.groups()
        year, month, day = int(year), int(month), int(day)

        if not (1 <= day <= 31) or not (1 <= month <= 12):
            return None

        try:
            return datetime(year, month, day)
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # TIME PARSING (handles multiple OCR formats)
    # ------------------------------------------------------------------
    def _parse_time(self, time_str):
        if not time_str:
            return None

        try:
            cleaned = time_str.strip()
            cleaned = cleaned.replace(".", ":").replace(";", ":").replace(",", ":")
            # normalize A:M / P:M back to AM/PM after dot replacement
            cleaned = cleaned.replace("A:M", "AM").replace("P:M", "PM")
            dt = dateutil_parser.parse(cleaned, default=dateutil_parser.parse("00:00"))
            return (dt.hour, dt.minute)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # CHRONOLOGY VALIDATION (handles overnight shifts)
    # ------------------------------------------------------------------
    def _is_chronological(self, in_time, out_time):
        in_minutes = in_time[0] * 60 + in_time[1]
        out_minutes = out_time[0] * 60 + out_time[1]

        if in_minutes == out_minutes:
            return False

        if in_minutes < out_minutes:
            return True

        overnight_gap = (1440 - in_minutes) + out_minutes
        return overnight_gap <= self.MAX_OVERNIGHT_HOURS * 60
