from constants import STATUS_WORK, OFF_STATUSES


class AttendanceValidator:

    def validate(self, records):

        validated = []

        for record in records:

            status = record.get("status", "")
            in_time = record.get("in_time", "")
            out_time = record.get("out_time", "")

            # ----------------------------
            # Rule 1 & 2 : LEAVE / WEEK OFF
            # ----------------------------
            if status in OFF_STATUSES:
                record["in_time"] = ""
                record["out_time"] = ""

            # ----------------------------
            # Rule 3 : If times exist
            # but status missing
            # ----------------------------
            # No else branch for "no time, no status": leave status
            # blank even when a date is present. Only explicit
            # leave/weekoff text (caught earlier by StatusDetector)
            # should ever produce those statuses -- a row with a date
            # but nothing else filled in yet is not evidence of a
            # week off, just an unfilled row.
            elif status == "":
                if in_time or out_time:
                    record["status"] = STATUS_WORK

            validated.append(record)

        return validated