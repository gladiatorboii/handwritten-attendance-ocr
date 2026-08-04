import json
import os
import time
import fitz
import tempfile
import shutil
from PIL import Image, ImageOps

from preprocessing.deskew import deskew_image
from preprocessing.crop import crop_to_content, crop_header_region
from preprocessing.rotation import needs_rotation, rotate_image
from validation_engine import ValidationEngine
from post_processing import process_records
from excel_report import generate_excel_report
from validators import AttendanceValidator
from mistral_ocr_engine import MistralOCREngine
from vision_recheck_engine import VisionRecheckEngine
from recheck_utils import find_outlier_records, needs_recheck, disagrees
from constants import STATUS_WORK
from config import (
    PREPROCESSED_DIR,
    get_output_paths,
    MISTRAL_API_KEY,
    GROQ_API_KEY,
    debug_print
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
                "Qwen 3.6 27B (via Groq) is the row-recheck engine this "
                "pipeline uses to independently verify flagged rows."
            )

        self.validator = AttendanceValidator()
        self.validation_engine = ValidationEngine()
        self.mistral_ocr_engine = MistralOCREngine(api_key=MISTRAL_API_KEY)
        self.vision_recheck_engine = VisionRecheckEngine(api_key=GROQ_API_KEY)

    def _load_input(self, input_path):

        extension = os.path.splitext(input_path)[1].lower()

        # Image
        if extension != ".pdf":
            temp_dir = tempfile.mkdtemp(prefix="attendance_pages_")
            image_path = self._correct_rotation(input_path, temp_dir)
            pages = self.split_double_page(image_path, temp_dir)
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

            image_path = self._correct_rotation(image_path, temp_dir)

            pages.extend(self.split_double_page(image_path, temp_dir))

        doc.close()

        return pages, temp_dir

    def _correct_rotation(self, image_path, temp_dir):
        """
        Some scans have their content genuinely rotated 90 degrees from
        their own pixel dimensions -- confirmed on a real PDF where every
        page's embedded photo held a landscape two-page spread but was
        saved with portrait pixel dimensions, no EXIF orientation tag,
        and no PDF page /Rotate flag anywhere signaling this (so neither
        this method's own EXIF handling elsewhere, nor split_double_page's
        aspect-ratio check, ever had a signal to act on -- the image just
        silently stayed sideways, and Mistral could read some of it but
        unreliably lost a whole second employee's table on several pages
        that shape affected).

        needs_rotation() (Hough-line orientation analysis, see
        preprocessing/rotation.py) only detects THAT a 90-degree
        correction is likely needed, not which of the two directions is
        right -- getting that wrong would just trade one broken
        orientation for another (upside down instead of sideways). Both
        candidates get a real, cheap verification: crop just the header-
        sized top strip of each and run one OCR call on it, keeping
        whichever direction actually finds a name. Falls back to the
        original, uncorrected image if neither does (better to leave a
        page as it was than confidently apply a coin-flip rotation).
        """
        if not needs_rotation(image_path):
            return image_path

        base = os.path.splitext(os.path.basename(image_path))[0]
        best_path = image_path

        for degrees in (90, -90):
            candidate_path = os.path.join(temp_dir, f"{base}_rot{degrees}.jpg")
            rotate_image(image_path, degrees, candidate_path)

            probe_path = os.path.join(temp_dir, f"{base}_rot{degrees}_probe.jpg")
            crop_header_region(candidate_path, probe_path)
            probe_blocks = self.mistral_ocr_engine.run(probe_path)

            if any(b.get("signature_name") for b in probe_blocks):
                debug_print(
                    f"AttendancePipeline: rotated {degrees} degrees "
                    f"(confirmed via header probe)"
                )
                return candidate_path

        debug_print(
            "AttendancePipeline: image looked rotated but neither "
            "direction's header probe found a name -- leaving unrotated"
        )
        return best_path

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
                # occasionally two employees share one page (see
                # MistralOCREngine._EXTRACTION_SCHEMA) -- _process_page
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
        # Almost always one block; occasionally two employees share one
        # physical page, in which case each block already carries its own
        # name/code/records (see MistralOCREngine._EXTRACTION_SCHEMA) and
        # is processed through the rest of the pipeline independently,
        # returned as its own result.
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

        # A block missing its own name/code (the heading was too garbled
        # for the model to read alongside the full page) gets a second,
        # more focused look at just the header crop -- costs one extra
        # call only on the pages that actually need it. Matched back to
        # the block needing it by position; if the header crop returns
        # fewer entries than there are blocks (e.g. one employee's
        # heading was legible enough to already succeed), the first
        # header-crop entry is reused as the best available guess.
        if any(not b.get("signature_name") or not b.get("employee_code") for b in blocks):
            header_path = os.path.join(
                PREPROCESSED_DIR,
                f"header_{os.path.basename(image_path)}"
            )
            crop_header_region(deskewed_path, header_path)
            header_blocks = self.mistral_ocr_engine.run(header_path)

            if header_blocks:
                for i, b in enumerate(blocks):
                    if b.get("signature_name") and b.get("employee_code"):
                        continue
                    hb = header_blocks[i] if i < len(header_blocks) else header_blocks[0]
                    if not b.get("signature_name"):
                        b["signature_name"] = hb.get("signature_name", "")
                    if not b.get("employee_code"):
                        b["employee_code"] = hb.get("employee_code", "")

        ocr_time = time.perf_counter() - ocr_start

        results = []
        for block_index, block in enumerate(blocks):
            records = block.get("records", [])
            employee_name = block.get("signature_name", "")
            employee_code = block.get("employee_code", "")
            block_hint = self._block_hint(block_index, len(blocks)) if len(blocks) > 1 else None

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
        # Second-Opinion Recheck (independent second opinion on suspicious rows)
        #
        # Gated by recheck_utils' engine-agnostic outlier/format check
        # rather than ValidationEngine's cell_highlight -- recheck has to
        # run before validate_records() so its disagreement flags can
        # feed into that final flagging decision (see
        # vision_recheck_engine.py docstring for why an independent model,
        # not Mistral re-reading itself, is used here -- and for why that
        # model itself has already had to be swapped once).
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

        # Collect every row/field that needs a second look first, rather
        # than calling the recheck engine as each one is found -- see
        # vision_recheck_engine.py's docstring for why one call per row
        # blew through this model's per-minute token quota on a real
        # page. All of them go to the engine together in one call below.
        recheck_targets = []

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

            fields_needed = [
                field for field in ("in_time", "out_time")
                if needs_recheck(record, field, outlier_ids)
            ]
            if fields_needed:
                recheck_targets.append((record, date, disambiguator, fields_needed))

        if recheck_targets:
            batch_requests = [
                (date, disambiguator) for _, date, disambiguator, _ in recheck_targets
            ]
            batch_results = self.vision_recheck_engine.recheck_rows(
                deskewed_path, batch_requests
            )

            if batch_results is not None:
                for (record, date, disambiguator, fields_needed), recheck in zip(
                    recheck_targets, batch_results
                ):
                    for field in fields_needed:
                        recheck_value = recheck.get(field, "") or ""
                        record[f"{field}_recheck_value"] = recheck_value
                        record[f"{field}_ocr_disagreement"] = disagrees(
                            record.get(field, ""), recheck_value
                        )

        timings["Second-Opinion Recheck"] = time.perf_counter() - start
        self._log_stage("Second-Opinion Recheck", timings["Second-Opinion Recheck"])

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
