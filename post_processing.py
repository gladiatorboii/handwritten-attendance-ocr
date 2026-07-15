import re
from dateutil import parser as dateutil_parser

from constants import STATUS_WORK, STATUS_LEAVE, STATUS_WOFF
from config import debug_print

def normalize_date(date_str):
    """
    Output is YYYY-MM-DD -- registers themselves are always written
    DD/MM/YY(YY) (confirmed across every sample this project has seen),
    so parsing still reads the source in that order; only the returned
    string's field order, separator, and year width change.
    """
    if not date_str:
        return ""

    value = (
        str(date_str)
        .strip()
        .replace("|", "/")
        .replace("\\", "/")
        .replace("-", "/")
        .replace(" ", "")
    )

    # Keep only digits and '/'
    value = re.sub(r"[^0-9/]", "", value)

    parts = value.split("/")

    if len(parts) == 3:
        day, month, year = parts
    else:
        # Handle OCR that drops the separators entirely, e.g. "01062026"
        # or "010626"
        digits = re.sub(r"\D", "", value)

        if len(digits) == 8:
            day, month, year = digits[:2], digits[2:4], digits[4:8]
        elif len(digits) == 6:
            day, month, year = digits[:2], digits[2:4], digits[4:6]
        else:
            return value

    day = day.zfill(2)
    month = month.zfill(2)

    # Convert a 1- or 2-digit year to a full 4-digit one (26 -> 2026);
    # a 4-digit year is left as-is. A 3-digit year is a stray extra
    # digit Mistral occasionally inserts (confirmed on real OCR output:
    # "03-06-026" for a handwritten "03-06-26") -- the last 2 characters
    # are the real 2-digit year regardless of what came before them.
    if len(year) == 1:
        year = year.zfill(2)
    elif len(year) == 3:
        year = year[-2:]

    if len(year) == 2:
        year = "20" + year

    return f"{year}-{month}-{day}"


def normalize_time(time_str):
    from dateutil import parser as dateutil_parser

    if not time_str:
        return ""

    try:
        cleaned = str(time_str).strip()
        # A handwritten colon sometimes gets OCR'd as an apostrophe right
        # before the real separator ("6'.50" for "6:50") rather than
        # being recognized as a colon outright -- dropped outright since
        # it's not meant to be a separator itself, just noise next to one.
        cleaned = cleaned.replace("'", "").replace("’", "").replace("‘", "")
        # Normalize A.M / P.M before dot replacement
        cleaned = cleaned.replace("A.M", "AM").replace("P.M", "PM")
        cleaned = cleaned.replace("a.m", "AM").replace("p.m", "PM")
        cleaned = cleaned.replace(".", ":").replace(",", ":").replace(";", ":").replace(" ", "")
        # Fix A:M / P:M after dot replacement
        cleaned = cleaned.replace("A:M", "AM").replace("P:M", "PM")
        cleaned = re.sub(r'(\d)(AM|PM)', r'\1 \2', cleaned)

        # If hour > 12, it's already 24hr — strip AM/PM marker
        cleaned = re.sub(r'^(\d{1,2}:\d{2})\s*(AM|PM)$', 
                         lambda m: m.group(1) if int(m.group(1).split(':')[0]) > 12 else m.group(0), 
                         cleaned)
        dt = dateutil_parser.parse(cleaned, default=dateutil_parser.parse("00:00"))
        debug_print(f"DEBUG normalize_time: '{time_str}' -> cleaned='{cleaned}' -> {dt.hour:02}:{dt.minute:02}")
        return f"{dt.hour:02}:{dt.minute:02}"

    except Exception:
        # Some handwritten colons get OCR'd as a stray "1" digit instead
        # of a separator (seen consistently for one employee's writing
        # style: "18:51" -> "18151", "18:44" -> "18144") -- if the
        # cleaned string is otherwise all digits with a lone "1" sitting
        # where HH:MM's separator would go, try treating that "1" as the
        # colon before giving up.
        colon_fix_match = re.fullmatch(r'(\d{1,2})1(\d{2})', cleaned)

        # Same handwriting-colon ambiguity, different shape: a stray "1"
        # right before the (already period->colon-converted) separator,
        # e.g. "10:35" written in a way that reads as "101.35" -> cleaned
        # to "101:35" by the substitution above, rather than swallowing
        # the separator outright like the case above.
        if not colon_fix_match:
            colon_fix_match = re.fullmatch(r'(\d{1,2})1:(\d{2})', cleaned)

        if colon_fix_match:
            try:
                dt = dateutil_parser.parse(
                    f"{colon_fix_match.group(1)}:{colon_fix_match.group(2)}",
                    default=dateutil_parser.parse("00:00"),
                )
                debug_print(
                    f"DEBUG normalize_time: '{time_str}' -> colon-as-1 fix "
                    f"-> {dt.hour:02}:{dt.minute:02}"
                )
                return f"{dt.hour:02}:{dt.minute:02}"
            except Exception:
                pass
        return str(time_str).strip()


def normalize_status(status):

    if not status:
        return ""

    status = str(status).strip().upper()

    mapping = {
        STATUS_WORK: STATUS_WORK,
        STATUS_LEAVE: STATUS_LEAVE,
        STATUS_WOFF: STATUS_WOFF,
        "W/OFF": STATUS_WOFF,
        "WEEK OFF": STATUS_WOFF,
        "WEEKOFF": STATUS_WOFF,
        "OFF": STATUS_WOFF
    }

    return mapping.get(status, status)


def clean_record(record):

    return {
        "date": normalize_date(
            record.get("date")
        ),

        "in_time": normalize_time(
            record.get("in_time")
        ),

        "out_time": normalize_time(
            record.get("out_time")
        ),

        "status": normalize_status(
            record.get("status")
        ),

        "remark": record.get("remark", ""),

        "struck_through": bool(record.get("struck_through"))
    }


def process_records(records):
    return [clean_record(record) for record in records]