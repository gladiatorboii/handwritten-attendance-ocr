import re
import statistics

from constants import STATUS_WORK

# A real time cell, in strict HH:MM 24-hour form. Anything else (blank,
# garbled, a status word) fails this and is treated as worth a second
# read on its own, regardless of outlier status.
_STRICT_TIME_FORMAT = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")

MIN_SAMPLES_FOR_STATS = 5
MAD_THRESHOLD = 3.5


def find_outlier_records(records):
    """
    Rows whose in/out time is a statistical outlier against the rest of
    this page's WORK rows (MAD-based), independent of whichever engine
    produced the records or whichever engine will perform the recheck --
    this only looks at the record dicts.
    """
    durations, in_minutes, out_minutes = [], [], []

    for record in records:
        if record.get("status") != STATUS_WORK:
            continue

        in_t = _to_minutes(record.get("in_time", ""))
        out_t = _to_minutes(record.get("out_time", ""))

        if in_t is not None:
            in_minutes.append((record, in_t))
        if out_t is not None:
            out_minutes.append((record, out_t))

        if in_t is not None and out_t is not None:
            duration = out_t - in_t if out_t >= in_t else (1440 - in_t) + out_t
            durations.append((record, duration))

    outliers = set()
    outliers |= _mad_outliers(durations)
    outliers |= _mad_outliers(in_minutes)
    outliers |= _mad_outliers(out_minutes)
    return outliers


def needs_recheck(record, field, outlier_ids):
    value = record.get(field, "")

    if not value:
        return False  # nothing there to double-check

    if not _STRICT_TIME_FORMAT.match(value):
        return True  # doesn't even look like a time -- worth a second read

    return id(record) in outlier_ids


def disagrees(original_time, recheck_time):
    # Compares the full HH:MM, not just the hour -- a recheck re-reads
    # the whole page, so minute-level disagreement between two
    # independent reads is real signal the digit itself is ambiguous
    # (e.g. a "58" written over/close enough to double as "52"), not
    # scan-quality jitter.
    original_minutes = _to_minutes(original_time)
    recheck_minutes = _to_minutes(recheck_time)

    if original_minutes is None or recheck_minutes is None:
        return False

    return original_minutes != recheck_minutes


def _mad_outliers(pairs):
    if len(pairs) < MIN_SAMPLES_FOR_STATS:
        return set()

    values = [v for _, v in pairs]
    median = statistics.median(values)
    mad = statistics.median(abs(v - median) for v in values)

    if mad == 0:
        return set()

    flagged = set()
    for record, v in pairs:
        modified_z = 0.6745 * (v - median) / mad
        if abs(modified_z) >= MAD_THRESHOLD:
            flagged.add(id(record))
    return flagged


def _to_minutes(value):
    if not value or not _STRICT_TIME_FORMAT.match(value):
        return None
    hour, minute = value.split(":")
    return int(hour) * 60 + int(minute)
