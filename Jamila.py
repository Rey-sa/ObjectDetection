import cv2
import numpy as np


# --------------------
# functions
# --------------------
def create_color_masks(hsv, color_ranges):
    masks = {}
    for color, ranges in color_ranges.items():
        mask = None
        for r in ranges:
            lower, upper = np.array(r[0]), np.array(r[1])
            m = cv2.inRange(hsv, lower, upper)
            mask = m if mask is None else cv2.bitwise_or(mask, m)
        masks[color] = mask
    return masks


def classify_shape(cnt):
    area = cv2.contourArea(cnt)
    if area < 800:
        return None

    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
    circularity = 4 * np.pi * (area / (peri * peri + 1e-6))

    # --- Shape recognition ---
    if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        ratio = max(w, h) / max(1, min(w, h))
        if ratio <= 1.2:
            return "Wuerfel"
        else:
            return "Quader"
    elif circularity > 0.6 and len(approx) >= 6:
        return "Zylinder"
    elif len(approx) >= 7 and circularity < 0.6:
        return "Figur"
    else:
        return None


def process_contours(frame, contour_dict):
    display = frame.copy()

    for color, contours in contour_dict.items():
        for cnt in contours:
            shape = classify_shape(cnt)
            if not shape:
                continue

            img_height, img_width = display.shape[:2]
            norm_x, norm_y = get_normalized_coordinates(cnt, img_width, img_height)

            area = cv2.contourArea(cnt)
            if area < 800:
                continue

            # Get Middle
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Draw around objects + put text
            cv2.drawContours(display, [cnt], -1, (0, 255, 0), 2)

            # first row
            cv2.putText(display,
                        f"{color}-{shape}",
                        (cx - 40, cy + 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1)

            # second row
            cv2.putText(display,
                        f"{norm_x:.2f} | {norm_y:.2f}",
                        (cx - 40, cy + 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1)

    return display


def get_normalized_coordinates(contour, img_width, img_height):
    # BoundingRect
    x, y, w, h = cv2.boundingRect(contour)

    # Middle of pixle coordinates
    cx = x + w / 2
    cy = y + h / 2

    # Normalisierung auf [0,1] in warped Bereich
    norm_x = cx / img_width
    norm_y = 1 - (cy / img_height)

    return norm_x, norm_y


# ------------------
# main videocapture loop
# ------------------
cap = cv2.VideoCapture(1)

color_ranges = {
    "red": [
        ([0, 120, 50], [10, 255, 255]),
        ([0, 120, 120], [10, 255, 255]),
        ([170, 120, 50], [180, 255, 255]),
        ([170, 120, 255], [180, 255, 255])
    ],
    "green": [([35, 40, 40], [85, 255, 255])],
    "blue": [([95, 100, 70], [130, 255, 255])],
    "yellow": [([18, 50, 40], [35, 255, 255])]
}

if not cap.isOpened():
    raise IOError("Kamera kann nicht geöffnet werden.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    biggest_quad = None
    biggest_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 5000:  # ignore small contours
            continue

        # simpelize contours
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        # only if its a scare
        if len(approx) == 4:
            if area > biggest_area:
                biggest_area = area
                biggest_quad = approx

    display = frame.copy()
    display_warped = None

    # draw around 18x18 area
    if biggest_quad is not None:
        pts = biggest_quad.reshape(4, 2).astype(np.float32)

        # sort points (top left -> top right -> bottom left -> bottom right)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        rect = np.round(rect, 1)

        # draw square
        cv2.drawContours(display, [biggest_quad], -1, (0, 255, 0), 3)

        # transform corner points
        trns_pts = np.array([
            [0, 0],
            [400 - 1, 0],
            [400 - 1, 400 - 1],
            [0, 400 - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, trns_pts)

        warped = cv2.warpPerspective(frame, M, (400, 400))

        display_warped = warped.copy()

        # color masks
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        masks = create_color_masks(hsv, color_ranges)

        red_mask = masks["red"]
        green_mask = masks["green"]
        blue_mask = masks["blue"]
        yellow_mask = masks["yellow"]

        # contours
        contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_green, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_blue, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_yellow, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contour_dict = {
            "red": contours_red,
            "green": contours_green,
            "blue": contours_blue,
            "yellow": contours_yellow
        }

        display_warped = process_contours(warped, contour_dict)

    cv2.imshow("original", frame)

    if display_warped is not None:
        cv2.imshow("Objekt-Erkennung", display_warped)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()