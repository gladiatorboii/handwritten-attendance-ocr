"""
Canonical attendance status codes, shared across extraction, validation,
and reporting so the vocabulary lives in exactly one place.
"""

STATUS_WORK = "Present"
STATUS_LEAVE = "LEAVE"
STATUS_WOFF = "Weekoff"

# Statuses that represent a day with no attendance times expected.
OFF_STATUSES = {STATUS_LEAVE, STATUS_WOFF}
