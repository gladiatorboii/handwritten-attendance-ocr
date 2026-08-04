import cv2
import numpy as np
from PIL import Image

# A page's own ruled grid has far more horizontal lines (one per row --
# a month's register runs 25-30+ rows) than vertical ones (a handful of
# column dividers). When a page's content is genuinely rotated 90
# degrees sideways relative to its own pixel dimensions -- confirmed on
# a real PDF where every page's embedded photo was saved with portrait
# pixel dimensions despite holding a landscape two-page spread, with no
# EXIF orientation tag and no PDF page /Rotate flag anywhere signaling
# this (Mistral could still partly read it sideways, but unreliably
# lost a whole second employee's table on several pages) -- that same
# grid's many short evenly-spaced lines show up as near-vertical instead
# of near-horizontal. Comparing which orientation dominates catches this
# independent of any metadata, since none exists for this failure mode.
VERTICAL_DOMINANCE_RATIO = 1.5


def needs_rotation(image_path):
    """
    True if this image's own ruled/grid lines are predominantly vertical
    rather than horizontal -- evidence the whole image is rotated 90
    degrees from its intended reading orientation. Direction (which way
    to rotate) isn't decided here -- see pipeline.py's caller, which
    tries both and keeps whichever one a real extraction call confirms.
    """
    image = cv2.imread(image_path)
    if image is None:
        return False

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=100, minLineLength=300, maxLineGap=20
    )

    if lines is None:
        return False

    horizontal = 0
    vertical = 0

    for line in lines:
        coords = np.asarray(line).reshape(-1)
        if len(coords) != 4:
            continue
        x1, y1, x2, y2 = coords
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle < 20 or angle > 160:
            horizontal += 1
        elif 70 < angle < 110:
            vertical += 1

    if horizontal == 0:
        return vertical > 0

    return vertical / horizontal >= VERTICAL_DOMINANCE_RATIO


def rotate_image(image_path, degrees, output_path):
    """
    Rotates the whole image by `degrees` (positive = counterclockwise,
    matching PIL's own rotate() convention) and saves it. `expand=True`
    so a 90-degree rotation swaps width/height instead of cropping into
    the rotated content.
    """
    with Image.open(image_path) as img:
        rotated = img.rotate(degrees, expand=True)
        rotated.convert("RGB").save(output_path)
    return output_path
