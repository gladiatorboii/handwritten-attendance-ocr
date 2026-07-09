import cv2
import numpy as np

from config import debug_print


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

    if len(angles) == 0:

        cv2.imwrite(
            output_path,
            image
        )

        return output_path

    skew_angle = np.median(angles)

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