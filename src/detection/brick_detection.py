import cv2
from src.config.settings import DESTINATION_SIZE, FIELD_CM, STUD_CM

def analyze_bricks(color, contours, warp):
    cm_per_pixel = FIELD_CM / DESTINATION_SIZE
    bricks = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue
        x, y, width, height = cv2.boundingRect(cnt)
        cx, cy = x + width // 2, y + height // 2

        nx = cx / DESTINATION_SIZE
        ny = 1 - (cy / DESTINATION_SIZE)

        w_cm, h_cm = width * cm_per_pixel, height * cm_per_pixel
        cols = max(1, round(w_cm / STUD_CM))
        rows = max(1, round(h_cm / STUD_CM))
        brick_type = f"{cols}x{rows}"

        bricks.append({
            "color": color,
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "nx": nx,
            "ny": ny,
            "type": brick_type
        })

    return bricks