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

    cell_highlight = record.get("cell_highlight", {})
    row_fully_invalid = record.get("row_fully_invalid", False)

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

    date_cls = 'class="cell-invalid"' if cell_highlight.get("date") else ""
    in_cls = 'class="cell-invalid"' if cell_highlight.get("in_time") else ""
    out_cls = 'class="cell-invalid"' if cell_highlight.get("out_time") else ""
    status_cls = 'class="cell-invalid"' if cell_highlight.get("status") else ""

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
        <h3>Employee Name : {employee}</h3>
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
        total_records = employee_data.get("total_records", 0)
        records = employee_data.get("records", [])

        rows = "".join(_build_row(r) for r in records)

        sections += f"""
        <div class="employee-section">
            <h3>Page {page} &nbsp;|&nbsp; Employee : {employee} &nbsp;|&nbsp; Total Records : {total_records}</h3>
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