import re
from rapidfuzz import fuzz

from constants import STATUS_WORK, STATUS_LEAVE, STATUS_WOFF


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

        # ---------- WORK ----------
        if row.get("in_time") or row.get("out_time"):
            return STATUS_WORK

        return ""