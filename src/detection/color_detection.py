import cv2
import numpy as np
from src.config.settings import COLOR_RANGES, WHITE_MASK_RANGE


def detect_colors(hsv):
    results = []
    kernel = np.ones((7, 7), np.uint8)
    blurred = cv2.GaussianBlur(hsv, (5, 5), 0)

    red_masks = []

    for name, (lower, upper) in COLOR_RANGES.items():
        lower = np.array(lower, np.uint8)
        upper = np.array(upper, np.uint8)

        mask = cv2.inRange(blurred, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)

        colored_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        cv2.imshow(f"mask_{name}", colored_mask)


        # Wenn es "red" oder "red2" heißt → zu kombinierter Rotmaske hinzufügen
        if name.startswith("red"):
            red_masks.append(mask)
        else:
            # Normale Farbe direkt hinzufügen
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            results.append((name, contours))

    # Falls zwei Rotmasken existieren → kombinieren
    if len(red_masks) > 0:
        combined_red = red_masks[0]
        for m in red_masks[1:]:
            combined_red = cv2.bitwise_or(combined_red, m)

        contours_red, _ = cv2.findContours(combined_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results.append(("red", contours_red))

    return results
