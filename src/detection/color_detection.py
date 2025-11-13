import cv2
import numpy as np
from src.config.settings import COLOR_RANGES

def detect_colors(hsv):
    results = []
    kernel = np.ones((7, 7), np.uint8)
    for color, (lower, upper) in COLOR_RANGES.items():

        lower = np.array(lower, np.uint8)
        upper = np.array(upper, np.uint8)

        mask = cv2.inRange(hsv, lower, upper)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results.append((color, contours))

    return results
