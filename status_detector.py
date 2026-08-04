import re
from rapidfuzz import fuzz

from constants import STATUS_WORK, STATUS_LEAVE, STATUS_WOFF

# Mistral sometimes only transcribes part of a handwritten weekoff marker
# and drops the rest -- confirmed on real pages in two different shapes:
# a lone cell reading just "WEE" (from "WEEKLY OFF"/"WEEK OFF", the first
# syllable only), and "W/O" (from "W/OFF", dropping the second "F"). Both
# are too short for fuzzy_search's partial_ratio to reliably clear the
# WOFF threshold against the full keyword list, so they need their own
# check: the *entire* cell (not a substring of a longer word), stripped
# to bare letters, must be one of these exact truncations.
_WOFF_FRAGMENT = re.compile(r"^(?:wee[a-z]*|wo)$", re.IGNORECASE)


class StatusDetector:

    STATUS_PATTERNS = {

        STATUS_LEAVE: [
            "leave", "leavs", "leav", "leaue", "1eave",
            "holiday", "holyday", "holidey"
        ],

        STATUS_WOFF: [
            "woff", "w/off", "w off", "w.off", "w/oft",
            "week off", "weekly off", "weeklyoff", "wk off",
            "weekoff", "wloff", "wlot", "w|off", "wooff",
            "oloff", "w/0ff", "woff", "off"
        ]

    }

    def normalize(self, text):
        text = text.lower()
        text = re.sub(r"[./\-|\\]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def fuzzy_search(self, row_text, keywords, threshold=85):
        normalized_row = self.normalize(row_text)
        for keyword in keywords:
            score = fuzz.partial_ratio(
                self.normalize(keyword),
                normalized_row
            )
            if score >= threshold:
                return True
        return False

    def _all_row_text(self, row):
        """Collect text from every possible field in the row."""
        parts = list(row.get("cells", []))
        for field in ("in_time", "out_time", "date"):
            val = row.get(field, "") or ""
            if val:
                parts.append(val)
        return " ".join(parts)

    def detect(self, row):
        # ---------- WORK ----------
        # Checked first, ahead of the LEAVE/WOFF text search below --
        # confirmed on a real page where a stray "W/Off" sitting in an
        # unrelated trailing column (noise bled in from an adjacent
        # cell, not this row's own status) overrode a row that had
        # perfectly real clocked times, wrongly labeling a worked day as
        # a week off. A real in_time/out_time means the person was
        # there that day -- full stop, regardless of what other leave/
        # weekoff-looking text happens to appear elsewhere in the row.
        if row.get("in_time") or row.get("out_time"):
            return STATUS_WORK

        row_text = self._all_row_text(row)

        # ---------- LEAVE ----------
        if self.fuzzy_search(
            row_text,
            self.STATUS_PATTERNS[STATUS_LEAVE],
            88
        ):
            return STATUS_LEAVE

        # ---------- WOFF ----------
        if self.fuzzy_search(
            row_text,
            self.STATUS_PATTERNS[STATUS_WOFF],
            82
        ):
            return STATUS_WOFF

        # ---------- WOFF (truncated fragment, whole cell only) ----------
        for cell in row.get("cells", []):
            token = re.sub(r"[^A-Za-z]", "", cell or "")
            if token and _WOFF_FRAGMENT.match(token):
                return STATUS_WOFF

        return ""