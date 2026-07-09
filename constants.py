"""
Canonical attendance status codes, shared across extraction, validation,
and reporting so the vocabulary lives in exactly one place.
"""

STATUS_WORK = "WORK"
STATUS_LEAVE = "LEAVE"
STATUS_WOFF = "WOFF"

# Statuses that represent a day with no attendance times expected.
OFF_STATUSES = {STATUS_LEAVE, STATUS_WOFF}
