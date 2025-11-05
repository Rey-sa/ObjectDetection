import cv2
import cv2 as cv
import numpy as np
from .get_limits import get_limits

COLOR_MAP = {
    "RED": [0, 0, 255],
    "GREEN": [0, 255, 0],
    "BLUE": [255, 0, 0],
    "YELLOW": [0, 255, 255]
}

def detect_color(roi_bgr):
    """
    Detects dominant colors in an videostream based on HSV color ranges.

    Returns:
        detected_color: String or unknown
    """
    hsv = cv.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    max_pixels = 0
    detected_color = "Unknown"

    for name, bgr in COLOR_MAP.items():
        lower, upper = get_limits(bgr)
        mask = cv2.inRange(hsv, lower, upper)
        pixels = cv2.countNonZero(mask)

        if pixels > max_pixels:
            max_pixels = pixels
            detected_color = name

    return detected_color

