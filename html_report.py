def generate_html_report(data):
    if "employees" in data:
        return generate_multi_employee_html(data)
    return generate_single_employee_html(data)


def _build_row(record):
    status = record.get("status", "")
    in_time = (record.get("in_time", "") or "").strip()
    out_time = (record.get("out_time", "") or "").strip()
    date = record.get("date", "")
    remark = (record.get("remark", "") or "").strip()

    # cell_highlight is a list of just the flagged cell names (not a
    # 4-key dict spelling out "false" for every cell that isn't flagged
    # -- see validation_engine.ValidationEngine._decide_highlights).
    # row_fully_invalid isn't a separate field anymore either -- derived
    # here instead, since "all 4 cells flagged" is exactly what that
    # field always meant.
    cell_highlight = record.get("cell_highlight", [])
    row_fully_invalid = len(cell_highlight) == 4

    if row_fully_invalid:
        # Every cell is bad — highlight the whole row instead of
        # repeating the highlight class on each individual td.
        return f"""
        <tr class="row-invalid">
            <td>{date}</td>
            <td>{in_time}</td>
            <td>{out_time}</td>
            <td>{status}</td>
            <td>{remark}</td>
        </tr>
        """

    date_cls = 'class="cell-invalid"' if "date" in cell_highlight else ""
    in_cls = 'class="cell-invalid"' if "in_time" in cell_highlight else ""
    out_cls = 'class="cell-invalid"' if "out_time" in cell_highlight else ""
    status_cls = 'class="cell-invalid"' if "status" in cell_highlight else ""

    return f"""
        <tr>
            <td {date_cls}>{date}</td>
            <td {in_cls}>{in_time}</td>
            <td {out_cls}>{out_time}</td>
            <td {status_cls}>{status}</td>
            <td>{remark}</td>
        </tr>
        """


def generate_single_employee_html(data):

    rows = "".join(_build_row(r) for r in data["records"])
    employee = data.get("employee_name") or "Not Detected"
    employee_code = data.get("employee_code") or ""

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Attendance Report</title>
        <style>
            body {{ font-family: Arial; margin: 20px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid black; padding: 8px; text-align: center; min-width: 70px; }}
            td:empty::after {{ content: "\\00a0"; }}
            th {{ background: #f2f2f2; }}
            .row-invalid {{ background-color: #ffb3b3; font-weight: bold; }}
            .cell-invalid {{ background-color: #ffb3b3; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h2>Attendance Register Extraction</h2>
        <h3>Employee Name : {employee}{f" (Code: {employee_code})" if employee_code else ""}</h3>
        <p>Total Records : {data['total_records']}</p>
        <table>
            <tr>
                <th>Date</th>
                <th>In Time</th>
                <th>Out Time</th>
                <th>Status</th>
                <th>Remark</th>
            </tr>
            {rows}
        </table>
    </body>
    </html>
    """


def generate_multi_employee_html(data):

    total_pages = data.get("total_pages", 0)
    total_employees = data.get("total_employees", 0)

    sections = ""

    for employee_data in data.get("employees", []):

        page = employee_data.get("page", "")
        employee = employee_data.get("employee_name") or "Not Detected"
        employee_code = employee_data.get("employee_code") or ""
        total_records = employee_data.get("total_records", 0)
        records = employee_data.get("records", [])

        rows = "".join(_build_row(r) for r in records)

        code_suffix = f" (Code: {employee_code})" if employee_code else ""
        sections += f"""
        <div class="employee-section">
            <h3>Page {page} &nbsp;|&nbsp; Employee : {employee}{code_suffix} &nbsp;|&nbsp; Total Records : {total_records}</h3>
            <table>
                <tr>
                    <th>Date</th>
                    <th>In Time</th>
                    <th>Out Time</th>
                    <th>Status</th>
                    <th>Remark</th>
                </tr>
                {rows}
            </table>
        </div>
        <hr>
        """

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Attendance Report</title>
        <style>
            body {{ font-family: Arial; margin: 20px; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 10px; }}
            th, td {{ border: 1px solid black; padding: 8px; text-align: center; min-width: 70px; }}
            td:empty::after {{ content: "\\00a0"; }}
            th {{ background: #f2f2f2; }}
            .row-invalid {{ background-color: #ffb3b3; font-weight: bold; }}
            .cell-invalid {{ background-color: #ffb3b3; font-weight: bold; }}
            .employee-section {{ margin-bottom: 40px; }}
            hr {{ border: 2px solid #ccc; margin: 30px 0; }}
            .summary {{ background: #e8f4f8; padding: 15px; border-radius: 5px; margin-bottom: 30px; }}
        </style>
    </head>
    <body>
        <h2>Attendance Register Extraction</h2>
        <div class="summary">
            <p><strong>Total Pages :</strong> {total_pages} &nbsp;&nbsp;
               <strong>Total Employees :</strong> {total_employees}</p>
        </div>
        {sections}
    </body>
    </html>
    """