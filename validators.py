from constants import STATUS_WORK, STATUS_WOFF, OFF_STATUSES


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
            elif status == "":
                if in_time or out_time:
                    record["status"] = STATUS_WORK

                # ----------------------------
                # Rule 4 : No time, no status
                # date exists → WOFF
                # ----------------------------
                else:
                    if record.get("date", ""):
                        record["status"] = STATUS_WOFF

            validated.append(record)

        return validated