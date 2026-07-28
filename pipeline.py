import json
import os
import time
import fitz
import tempfile
import shutil
from PIL import Image, ImageOps

from preprocessing.deskew import deskew_image
from preprocessing.crop import crop_to_content, crop_header_region
from validation_engine import ValidationEngine
from post_processing import process_records
from excel_report import generate_excel_report
from validators import AttendanceValidator
from mistral_ocr_engine import MistralOCREngine
from llama_vision_engine import LlamaVisionEngine
from recheck_utils import find_outlier_records, needs_recheck, disagrees
from constants import STATUS_WORK
from config import (
    PREPROCESSED_DIR,
    get_output_paths,
    MISTRAL_API_KEY,
    GROQ_API_KEY
)


class AttendancePipeline:

    # A single portrait register page is taller than it is wide. A scan
    # noticeably wider than tall means the photo/scan caught two physical
    # pages of an open notebook at once (see split_double_page) rather
    # than one page -- some slack above 1.0 so a page that's merely
    # slightly wider due to camera angle/crop isn't split unnecessarily.
    DOUBLE_PAGE_ASPECT_RATIO = 1.15

    def __init__(self):

        if not MISTRAL_API_KEY:
            raise ValueError(
                "MISTRAL_API_KEY is not set (see .env / .env.example) -- "
                "Mistral is the primary OCR engine this pipeline has."
            )

        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set (see .env / .env.example) -- "
                "Llama 4 Scout (via Groq) is the row-recheck engine this "
                "pipeline uses to independently verify flagged rows."
            )

        self.validator = AttendanceValidator()
        self.validation_engine = ValidationEngine()
        self.mistral_ocr_engine = MistralOCREngine(api_key=MISTRAL_API_KEY)
        self.llama_vision_engine = LlamaVisionEngine(api_key=GROQ_API_KEY)

    def _load_input(self, input_path):

        extension = os.path.splitext(input_path)[1].lower()

        # Image
        if extension != ".pdf":
            temp_dir = tempfile.mkdtemp(prefix="attendance_pages_")
            pages = self.split_double_page(input_path, temp_dir)
            if len(pages) == 1 and pages[0] == input_path:
                self._cleanup(temp_dir)  # nothing was split/written, nothing to clean up
                return [input_path], None
            return pages, temp_dir

        # PDF
        doc = fitz.open(input_path)

        print(f"\nPDF detected.")
        print(f"Total Pages : {len(doc)}\n")

        temp_dir = tempfile.mkdtemp(prefix="attendance_pages_")

        pages = []

        for i in range(len(doc)):

            page = doc.load_page(i)

            pix = page.get_pixmap(
                matrix=fitz.Matrix(3, 3)
            )

            image_path = os.path.join(
                temp_dir,
                f"page_{i+1}.jpg"
            )

            pix.save(image_path)

            pages.extend(self.split_double_page(image_path, temp_dir))

        doc.close()

        return pages, temp_dir

    def split_double_page(self, image_path, temp_dir):
        """
        Some scans photograph two physical register pages at once (an
        open notebook spread) instead of one page per image --
        recognizable by the image being noticeably wider than it is
        tall, unlike a single portrait page (see DOUBLE_PAGE_ASPECT_RATIO).
        Splitting the image itself at the midpoint before OCR ever sees
        it avoids relying on Mistral's own table-structure detection
        (observed to be inconsistent across calls -- sometimes one wide
        table, sometimes two sequential tables, sometimes merged into
        one) to work out where one page ends and the next begins; each
        half becomes a completely independent single-table image
        instead, with no cross-page ambiguity left for recheck to
        stumble into.

        Returns [image_path] unchanged if the image isn't wide enough to
        be a double-page spread.

        Corrects the image's own EXIF rotation tag before measuring it --
        confirmed on a real double-page spread (EXIF Orientation 8) whose
        raw pixel dimensions were portrait-shaped even though the actual
        photo was landscape, silently skipping the split below and
        forcing a single OCR call to read both pages' tables (and both
        employees' names) at once instead of one focused call per page.
        PIL/cv2 don't apply EXIF rotation on their own, so every
        downstream size/crop decision (here and in preprocessing.crop)
        would otherwise silently see the wrong orientation too.

        Checked via the raw EXIF Orientation tag itself, not by comparing
        pre/post-transpose image size -- a 180-degree rotation or a pure
        mirror flip (Orientation 2, 3, or 4) never changes width/height,
        so a size comparison alone would silently miss exactly those
        cases and only catch the 90/270-degree ones (5-8) that happen to
        swap dimensions.
        """
        with Image.open(image_path) as raw:
            orientation = raw.getexif().get(0x0112, 1)  # standard EXIF Orientation tag ID
            img = ImageOps.exif_transpose(raw) if orientation != 1 else raw
            width, height = img.size

            if width / height < self.DOUBLE_PAGE_ASPECT_RATIO:
                if orientation == 1:
                    return [image_path]
                corrected_path = os.path.join(temp_dir, os.path.basename(image_path))
                img.convert("RGB").save(corrected_path)
                return [corrected_path]

            mid = width // 2
            left = img.crop((0, 0, mid, height)).convert("RGB")
            right = img.crop((mid, 0, width, height)).convert("RGB")

            base = os.path.splitext(os.path.basename(image_path))[0]
            left_path = os.path.join(temp_dir, f"{base}_left.jpg")
            right_path = os.path.join(temp_dir, f"{base}_right.jpg")

            left.save(left_path)
            right.save(right_path)

            return [left_path, right_path]

    def run(
        self,
        input_path,
        save_output=True,
        progress_callback=None
    ):

        total_start = time.perf_counter()

        pages, temp_dir = self._load_input(input_path)

        all_results = []

        try:

            for index, page_path in enumerate(pages, start=1):
                print("\n" + "=" * 60)
                print(f"Processing Page {index}/{len(pages)}")
                print("=" * 60)

                # Optional -- lets a caller (e.g. api.py's job tracker)
                # observe progress without this method needing to know
                # anything about how that caller reports it.
                if progress_callback:
                    progress_callback(index, len(pages))

                # A physical page is almost always one set of records, but
                # occasionally two full tables share one page (see
                # MistralOCREngine._find_group_ranges) -- _process_page
                # returns a list either way so neither case is special-cased
                # here.
                page_results = self.process_page(page_path)

                all_results.extend(page_results)

            if save_output:
                self._save_output(all_results, input_path)

            self.print_timing_summary(
                all_results,
                time.perf_counter() - total_start
            )

            if len(all_results) == 1:
                return all_results[0]

            return {
                "total_pages": len(all_results),
                "employees": all_results
            }

        finally:
            self._cleanup(temp_dir)


    def process_page(self, image_path):
        page_start = time.perf_counter()

        os.makedirs(PREPROCESSED_DIR, exist_ok=True)

        cropped_path = os.path.join(
            PREPROCESSED_DIR,
            f"cropped_{os.path.basename(image_path)}"
        )

        deskewed_path = os.path.join(
            PREPROCESSED_DIR,
            f"deskewed_{os.path.basename(image_path)}"
        )

        deskew_start = time.perf_counter()
        crop_to_content(image_path, cropped_path)
        deskew_image(cropped_path, deskewed_path)
        deskew_time = time.perf_counter() - deskew_start

        # -----------------------------
        # Mistral OCR (main, and only, extraction engine)
        #
        # Almost always one block; occasionally two full tables share one
        # physical page (see MistralOCREngine._find_group_ranges), in
        # which case each is processed through the rest of the pipeline
        # independently and returned as its own result.
        # -----------------------------
        ocr_start = time.perf_counter()

        blocks = self.mistral_ocr_engine.run(deskewed_path)

        if not blocks:
            # Confirmed by direct observation on a real page: preprocessing
            # can occasionally make an otherwise perfectly legible page
            # unreadable to Mistral even though the processed image looks
            # fine to the eye -- the original, untouched image OCR'd
            # correctly on the exact same page. Retrying with the
            # original before giving up on the page entirely costs one
            # extra call only on the rare page that comes back empty.
            blocks = self.mistral_ocr_engine.run(image_path)

        base_name = self.mistral_ocr_engine.extract_employee_name(
            self.mistral_ocr_engine.last_markdown
        )
        employee_code = self.mistral_ocr_engine.extract_employee_code(
            self.mistral_ocr_engine.last_markdown
        )

        # Mistral's markdown transcription sometimes garbles the heading
        # badly enough that neither field-extraction pass above finds
        # anything at all -- re-OCR'ing just the header crop (rather than
        # the whole page) gives Mistral a second, more focused look at
        # exactly that region, which costs one extra call only on the
        # pages that actually need it. Reuses the same extraction logic
        # already proven on the full page, just against a smaller image.
        if not base_name or not employee_code:
            header_path = os.path.join(
                PREPROCESSED_DIR,
                f"header_{os.path.basename(image_path)}"
            )
            crop_header_region(deskewed_path, header_path)
            self.mistral_ocr_engine.run(header_path)

            if not base_name:
                base_name = self.mistral_ocr_engine.extract_employee_name(
                    self.mistral_ocr_engine.last_markdown
                )
            if not employee_code:
                employee_code = self.mistral_ocr_engine.extract_employee_code(
                    self.mistral_ocr_engine.last_markdown
                )

        ocr_time = time.perf_counter() - ocr_start

        results = []
        for block_index, block in enumerate(blocks):
            records = block.get("records", [])
            signature_name = block.get("signature_name", "")

            employee_name = base_name
            block_hint = None
            if len(blocks) > 1:
                block_hint = self._block_hint(block_index, len(blocks))
                if self._names_look_different(base_name, signature_name):
                    # This block's own repeated signature is a genuinely
                    # different identity from the page-level name (see
                    # MistralOCREngine._extract_block_signature_name) --
                    # two different employees shared this page, so using
                    # the page-level name for both would silently
                    # mislabel one employee's records under the other's
                    # name.
                    employee_name = signature_name
                else:
                    label = f"table {block_index + 1}"
                    employee_name = f"{base_name} ({label})" if base_name else f"({label})"

            # Falls back to this block's own repeated signature (already
            # computed, no extra cost) when nothing above found a name at
            # all -- confirmed on a real page: the heading had no
            # separate name text of its own, but the sig column
            # consistently repeated the employee's name throughout the
            # table itself.
            if not employee_name and signature_name:
                employee_name = signature_name

            results.append(self._process_block(
                deskewed_path,
                records,
                employee_name,
                employee_code,
                block_hint,
                deskew_time,
                ocr_time,
                page_start
            ))

        return results

    def _names_look_different(self, base_name, signature_name):
        if not signature_name:
            return False
        if not base_name:
            return True
        return signature_name.lower() not in base_name.lower() and base_name.lower() not in signature_name.lower()

    def _block_hint(self, block_index, total_blocks):
        """
        Disambiguates which table a row-recheck call should look at when
        a page has more than one full attendance table side by side
        sharing the same dates (see MistralOCREngine.recheck_row).
        """
        if total_blocks == 2:
            return "in the left-hand table" if block_index == 0 else "in the right-hand table"
        return f"in table {block_index + 1} counting from the left"

    def _log_stage(self, stage, elapsed):
        print(f"{stage:<22}: {elapsed:>7.2f} sec")

    def _ordinal(self, n):
        if 11 <= n % 100 <= 13:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suffix}"

    def _process_block(
        self,
        deskewed_path,
        records,
        employee_name,
        employee_code,
        block_hint,
        deskew_time,
        ocr_time,
        page_start
    ):
        timings = {"Deskew": deskew_time, "Mistral OCR": ocr_time}

        print(f"\nEmployee : {employee_name}")
        print("-" * 60)
        self._log_stage("Deskew", deskew_time)
        self._log_stage("Mistral OCR", ocr_time)

        # -----------------------------
        # Post Processing
        # -----------------------------
        start = time.perf_counter()

        records = process_records(records)

        records = self.validator.validate(records)

        timings["Post Processing"] = time.perf_counter() - start
        self._log_stage("Post Processing", timings["Post Processing"])

        # -----------------------------
        # Llama Row Recheck (independent second opinion on suspicious rows)
        #
        # Gated by recheck_utils' engine-agnostic outlier/format check
        # rather than ValidationEngine's cell_highlight -- recheck has to
        # run before validate_records() so its disagreement flags can
        # feed into that final flagging decision (see
        # llama_vision_engine.py docstring for why an independent model,
        # not Mistral re-reading itself, is used here).
        # -----------------------------
        start = time.perf_counter()

        outlier_ids = find_outlier_records(records)

        # How many times each date appears in this block's own records --
        # an extra/overnight shift legitimately repeats a date, and
        # Mistral's table-structure rendering isn't fully deterministic
        # across calls, so a duplicate date can show up even when this
        # page's blocks weren't detected as split. Either way, "look at
        # the row with date X" is ambiguous once X repeats, so recheck
        # needs a position hint.
        date_totals = {}
        for record in records:
            d = record.get("date", "")
            if d:
                date_totals[d] = date_totals.get(d, 0) + 1
        date_seen = {}

        for record in records:
            if record.get("status") != STATUS_WORK:
                continue

            date = record.get("date", "")
            date_seen[date] = date_seen.get(date, 0) + 1

            disambiguator = block_hint
            if date_totals.get(date, 0) > 1:
                occurrence_hint = (
                    f"the {self._ordinal(date_seen[date])} row with this date, "
                    "counting from the top of the page"
                )
                disambiguator = f"{block_hint}, {occurrence_hint}" if block_hint else occurrence_hint

            for field in ("in_time", "out_time"):
                if not needs_recheck(record, field, outlier_ids):
                    continue

                recheck = self.llama_vision_engine.recheck_row(
                    deskewed_path, date, disambiguator
                )
                if recheck is None:
                    continue

                recheck_value = recheck.get(field, "") or ""
                record[f"{field}_llama_recheck"] = recheck_value
                record[f"{field}_ocr_disagreement"] = disagrees(
                    record.get(field, ""), recheck_value
                )

        timings["Llama Row Recheck"] = time.perf_counter() - start
        self._log_stage("Llama Row Recheck", timings["Llama Row Recheck"])

        # -----------------------------
        # Final Validation
        # -----------------------------
        start = time.perf_counter()

        records = self.validation_engine.validate_records(records)

        timings["Final Validation"] = time.perf_counter() - start
        self._log_stage("Final Validation", timings["Final Validation"])

        timings["Page Total"] = time.perf_counter() - page_start
        self._log_stage("Page Total", timings["Page Total"])
        print("-" * 60)

        return {
            "employee_name": employee_name,
            "employee_code": employee_code,
            "total_records": len(records),
            "records": records,
            "timings": timings
        }

    def _save_output(self, results, input_path):
        if len(results) == 1:
            structure = results[0]
        else:
            structure = {
                "total_pages": len(results),
                "total_employees": len(results),
                "employees": results
            }

        output_json, output_excel = get_output_paths(input_path)

        os.makedirs(os.path.dirname(output_json), exist_ok=True)

        with open(
            output_json,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                structure,
                f,
                indent=4
            )

        generate_excel_report(structure, output_excel)

    def _cleanup(self, temp_dir):
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


    def print_timing_summary(self, all_results, total_time):
        print("\n" + "=" * 60)
        print(" " * 17 + "PIPELINE TIMING SUMMARY")
        print("=" * 60)

        for entry_no, result in enumerate(all_results, start=1):
            print(f"\nEntry {entry_no}: {result.get('employee_name', '')}")
            print("-" * 60)

            for stage, elapsed in result["timings"].items():

                if stage == "Page Total":
                    continue

                print(f"{stage:<22}: {elapsed:>7.2f} sec")

            print("-" * 60)
            print(f"{'Page Total':<22}: {result['timings']['Page Total']:>7.2f} sec")

        print("\n" + "=" * 60)
        print(f"{'Overall Pipeline Time':<22}: {total_time:>7.2f} sec")
        print("=" * 60)
