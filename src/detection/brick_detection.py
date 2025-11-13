import cv2
from src.config.settings import DESTINATION_SIZE, FIELD_CM, STUD_CM

def analyze_bricks(color, contours, warp):
    cm_per_pixel = FIELD_CM / DESTINATION_SIZE
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2

        nx = cx / DESTINATION_SIZE
        ny = 1 - (cy / DESTINATION_SIZE)

        w_cm, h_cm = w * cm_per_pixel, h * cm_per_pixel
        cols = max(1, round(w_cm / STUD_CM))
        rows = max(1, round(h_cm / STUD_CM))
        brick_type = f"{cols}x{rows}"

        cv2.rectangle(warp, (x, y), (x + w, y + h), (0,255,255), 2)
        cv2.putText(warp, f"{color} {brick_type} ({nx:.2f},{ny:.2f})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
