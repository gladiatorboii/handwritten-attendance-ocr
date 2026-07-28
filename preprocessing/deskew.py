import shutil

import cv2
import numpy as np

from config import debug_print

# Below this angle, a rotation does more harm than good -- confirmed on
# a real page where the detected skew was only 0.91 degrees (essentially
# noise from slightly uneven table lines, not real photographic tilt),
# yet applying that rotation still introduced enough interpolation
# blur to make Mistral split a date across a phantom extra column. A
# genuinely tilted photo (5-15+ degrees from an unsteady hand) is well
# above this threshold and still gets corrected; a page that's already
# essentially straight no longer gets a risky rotation it doesn't need.
MIN_CORRECTION_DEGREES = 3.0


def deskew_image(input_path, output_path):

    image = cv2.imread(input_path)

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    edges = cv2.Canny(
        gray,
        50,
        150,
        apertureSize=3
    )

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=100,
        minLineLength=300,
        maxLineGap=20
    )

    angles = []

    if lines is not None:

        for line in lines:
            coords = np.asarray(line).reshape(-1)

            if len(coords) != 4:
                print(f"Unexpected HoughLinesP output: {coords}")
                continue

            x1, y1, x2, y2 = coords

            angle = np.degrees(
                np.arctan2(
                    y2 - y1,
                    x2 - x1
                )
            )

            if abs(angle) < 20:
                angles.append(angle)

    skew_angle = np.median(angles) if angles else 0.0

    if len(angles) == 0 or abs(skew_angle) < MIN_CORRECTION_DEGREES:

        # A plain file copy, not cv2.imread/imwrite's decode-and-re-encode
        # round-trip -- confirmed on a real page (see preprocessing/crop.py's
        # same fix) that the re-encode alone, even with nothing actually
        # transformed, can degrade the image just enough to make Mistral
        # misread tightly-spaced table columns.
        shutil.copy(input_path, output_path)

        return output_path

    h, w = image.shape[:2]

    center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(
        center,
        skew_angle,
        1.0
    )

    deskewed = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    cv2.imwrite(
        output_path,
        deskewed
    )

    debug_print(
        f"Detected skew angle: {skew_angle:.2f}"
    )

    return output_path
