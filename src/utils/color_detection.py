import cv2
import cv2 as cv
import numpy as np

COLOR_RANGES = {
    "RED": [(0, 120, 70), (10, 255, 255)],
    "RED2": [(170,120,70), (180, 255, 255)],
    "GREEN": [(36, 50, 70), (89, 255, 255)],
    "BLUE": [(90, 50, 70), (128, 255, 255)],
    "YELLOW": [(20, 100, 100), (35, 255, 255)],
    "ORANGE": [(10, 100, 100), (20, 255, 255)]
}

def detect_color(frame):
    """
    Detects dominant colors in an videostream based on HSV color ranges.

    Returns:
        dominant_color: String or unknown
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    detected_colors = {}

    for color_name, (lower, upper) in COLOR_RANGES.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        color_pixels = cv2.countNonZero(mask)
        detected_colors[color_name] = color_pixels

    # Red has 2 areas: Need to add
    total_red = detected_colors.get("RED",0) + detected_colors.get("RED2",0)
    detected_colors["RED"] = total_red
    detected_colors.pop("RED2", None)

    # Find dominant color
    dominant_color = max(detected_colors, key=detected_colors.get)

    if detected_colors[dominant_color] < 1000:
        return "Unknown"

    return dominant_color

