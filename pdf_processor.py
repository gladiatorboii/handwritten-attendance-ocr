import fitz
import os
import json
import time
import tempfile
import shutil

from excel_report import generate_excel_report
from config import get_output_paths


def process_pdf(pipeline, pdf_path, progress_callback=None):

    total_start = time.perf_counter()

    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    print(f"\nTotal Pages : {total_pages}\n")

    employees = []

    # A dedicated temp directory per call, not a fixed "outputs/temp_page_N"
    # name reused on every run -- two pipeline runs overlapping in time
    # (e.g. one kicked off before the previous one's background process
    # had finished) would otherwise silently overwrite each other's page
    # images mid-run, since the old fixed name gave concurrent runs no way
    # to avoid colliding on the exact same path.
    temp_dir = tempfile.mkdtemp(prefix="attendance_pdf_")

    try:
        for page_number in range(total_pages):

            print("=" * 50)
            print(f"Processing Page {page_number + 1}/{total_pages}")
            print("=" * 50)

            # Optional -- lets a caller (e.g. api.py's job tracker)
            # observe progress without this function needing to know
            # anything about how that caller reports it.
            if progress_callback:
                progress_callback(page_number + 1, total_pages)

            page = doc.load_page(page_number)
            pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))

            temp_image = os.path.join(
                temp_dir,
                f"temp_page_{page_number + 1}.png"
            )

            pix.save(temp_image)

            # Some scans have their content genuinely rotated 90 degrees
            # from their own pixel dimensions with no EXIF/PDF rotation
            # flag to signal it -- Mistral can still read some of a page
            # in that state but unreliably drops a whole second employee's
            # table. This mirrors AttendancePipeline._load_input's own PDF
            # branch exactly; this loop is a separate implementation (used
            # by both main.py and api.py for every real PDF run) that was
            # missing this call entirely, which is why the same
            # dropped-table symptom the rotation fix was built to solve
            # kept recurring here even after that fix was verified against
            # pipeline.run()'s PDF path.
            temp_image = pipeline._correct_rotation(temp_image, temp_dir)

            # Some scans photograph two physical register pages at once
            # (a landscape spread) -- split those into independent
            # single-page images before OCR ever sees them (see
            # AttendancePipeline.split_double_page) rather than relying
            # on Mistral's own, less consistent, multi-table detection.
            # Returns [temp_image] unchanged for an ordinary single page.
            sub_images = pipeline.split_double_page(temp_image, temp_dir)

            for sub_image in sub_images:
                # A physical (half-)page is almost always one employee's
                # records, but occasionally still holds two employees
                # side by side or in sequence (see MistralOCREngine's
                # _EXTRACTION_SCHEMA) -- process_page returns a list
                # either way, so every block gets its own entry instead
                # of silently keeping only one.
                page_results = pipeline.process_page(sub_image)

                for result in page_results:
                    employees.append({
                        "page": page_number + 1,
                        "employee_name": result.get("employee_name"),
                        "employee_code": result.get("employee_code"),
                        "total_records": result.get("total_records", 0),
                        "records": result.get("records", []),
                        "timings": result.get("timings", {})
                    })
    finally:
        doc.close()
        shutil.rmtree(temp_dir, ignore_errors=True)

    final_result = {
        "total_pages": total_pages,
        "total_employees": len(employees),
        "employees": employees
    }

    output_json, output_excel = get_output_paths(pdf_path)

    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(final_result, f, indent=4)

    generate_excel_report(final_result, output_excel)

    pipeline.print_timing_summary(employees, time.perf_counter() - total_start)

    print("\n" + "=" * 60)
    print(f"PDF Processing Completed — {len(employees)} employees processed")
    print("=" * 60)

    return final_result