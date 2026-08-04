import shutil

import cv2

from config import debug_print

# Below this fraction of the image's total area, the detected region is
# treated as noise rather than genuine page content -- cropping to it
# would very likely cut off real data.
MIN_CONTENT_AREA_FRACTION = 0.3

# If the detected boundary is already within this fraction of the full
# image on both axes, there's nothing meaningful to crop away.
MIN_REDUCTION_FRACTION = 0.03

# Extra margin kept around the detected content boundary, as a fraction
# of that boundary's own width/height -- the page's own ruled border can
# sit close to the true edge of the paper, so a tight crop with no
# padding risks slicing into real handwriting.
PADDING_FRACTION = 0.02

# A genuine page boundary is a quadrilateral (4 corners); this is how
# closely a contour's polygon approximation has to match that shape to
# be trusted as the page rather than an arbitrary blob of connected
# edges (shadows, desk grain, etc.).
MIN_POLYGON_VERTICES = 4
MAX_POLYGON_VERTICES = 6


# The employee name/code heading always sits in this top slice of the
# page, above the data table (confirmed across every sample this project
# has seen) -- generous enough to never cut off a heading that sits
# slightly lower, since a wider crop costs nothing but a little extra
# image for the recheck model to look past.
HEADER_REGION_FRACTION = 0.2


def crop_header_region(input_path, output_path):
    """
    Crops the top slice of a page image containing the employee
    name/code heading, so pipeline.py's header-recheck fallback (a
    second MistralOCREngine.run() call when the first pass finds no
    name/code) can look at just that region instead of the whole page.
    """
    image = cv2.imread(input_path)
    height = image.shape[0]
    cropped = image[: int(height * HEADER_REGION_FRACTION), :]
    cv2.imwrite(output_path, cropped)
    return output_path


def crop_to_content(input_path, output_path):
    """
    Crops away the background clutter around a photographed register
    page (desk surface, shadows, edges of other pages) instead of
    sending the whole photo to OCR. Brightness-based thresholding
    (Otsu) was tried first and rejected -- these are phone photos with
    uneven lighting/shadows across the page, so a single global
    brightness cutoff doesn't reliably separate paper from desk (it
    classified almost the entire page as "background" on a real test
    image). Edge detection looks for structure (the page's own straight
    boundary) rather than raw brightness, which isn't thrown off by
    lighting the same way: Canny finds edges, morphological closing
    bridges small gaps so the page's boundary forms one continuous
    contour instead of disconnected segments, and only a contour whose
    polygon approximation is roughly a quadrilateral (a page has 4
    sides) is trusted as the page rather than an arbitrary shadow/grain
    blob.

    Falls back to copying the image through unchanged if no contour
    both covers a plausible fraction of the frame and looks
    quadrilateral -- a failed/uncertain detection should never risk
    cropping into real content.
    """
    image = cv2.imread(input_path)
    image_h, image_w = image.shape[:2]
    image_area = image_h * image_w

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidate = None
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        if area < image_area * MIN_CONTENT_AREA_FRACTION:
            break  # sorted descending -- nothing after this is bigger

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if MIN_POLYGON_VERTICES <= len(approx) <= MAX_POLYGON_VERTICES:
            candidate = cv2.boundingRect(contour)
            break

    if candidate is None:
        # A plain file copy, not cv2.imread/imwrite's decode-and-re-encode
        # round-trip -- confirmed on a real page that the re-encode alone
        # (even with nothing actually cropped) degraded the image just
        # enough to make Mistral merge two adjacent table columns that
        # OCR'd as separate on the untouched original.
        debug_print("crop_to_content: no quadrilateral page-shaped contour found, skipping crop")
        shutil.copy(input_path, output_path)
        return output_path

    x, y, w, h = candidate

    if w >= image_w * (1 - MIN_REDUCTION_FRACTION) and h >= image_h * (1 - MIN_REDUCTION_FRACTION):
        debug_print("crop_to_content: detected boundary already fills the frame, skipping crop")
        shutil.copy(input_path, output_path)
        return output_path

    pad_x = int(w * PADDING_FRACTION)
    pad_y = int(h * PADDING_FRACTION)

    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(image_w, x + w + pad_x)
    y1 = min(image_h, y + h + pad_y)

    cropped = image[y0:y1, x0:x1]
    cv2.imwrite(output_path, cropped)

    return output_path
