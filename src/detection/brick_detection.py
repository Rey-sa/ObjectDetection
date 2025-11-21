import cv2
import numpy as np
from src.config.settings import DESTINATION_SIZE


def estimate_brick_type(cnt):
    rect = cv2.minAreaRect(cnt)
    width, height = rect[1]

    if width == 0 or height == 0:
        return "UNDEFINED"

    long_side = max(width, height)
    short_side = min(width, height)
    ratio = long_side / short_side

    if ratio < 1.2:
        return "1x1"
    elif ratio < 1.55:
        return "2x3"  # ratio ~1.40-1.45
    elif ratio < 1.78:
        return "1x2"  # ratio ~1.68-1.72
    elif ratio < 2.1:
        return "2x4"  # ratio ~1.84-1.85
    elif ratio < 2.6:
        return "1x3"  # ratio ~2.36-2.44
    else:
        return "UNDEFINED"

def analyze_bricks(color, contours, warp):
    """
    Analysis all contours of one color and returns a list of bricks.
    Every brick contains: color, x, y, width, height, nx, ny, type
    """
    bricks = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 200:
            continue

        # Bounding Boxes for position
        x, y, w, h = cv2.boundingRect(contour)
        cx, cy = x + w // 2, y + h // 2
        nx = cx / DESTINATION_SIZE
        ny = 1 - (cy / DESTINATION_SIZE)

        # Get type
        brick_type = estimate_brick_type(contour)

        bricks.append({
            "color": color,
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "nx": nx,
            "ny": ny,
            "type": brick_type
        })

    return bricks
