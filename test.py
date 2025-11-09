import cv2
import numpy as np

from src.camera.webcam import init_camera
from src.config.settings import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT

# --- Open camera---
cap = init_camera(CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT)

# --- HSV-Farbbereiche (ggf. anpassen) ---
color_ranges = {
    "red":   ([0, 120, 70], [10, 255, 255]),
    "green": ([35, 60, 60], [85, 255, 255]),
    "blue":  ([90, 80, 60], [130, 255, 255]),
    "yellow":([20, 100, 100], [30, 255, 255])
}

points = []
selecting = True

# --- Mausklicks für ROI-Auswahl ---
def mouse_click(event, x, y, flags, param):
    global points, selecting
    if event == cv2.EVENT_LBUTTONDOWN and selecting:
        points.append((x, y))
        print(f"Punkt {len(points)}: {x},{y}")

cv2.namedWindow("Kamera")
cv2.setMouseCallback("Kamera", mouse_click)

print("👉 Klicke die vier Ecken des 18×18 cm Felds im Uhrzeigersinn (oben-links start).")

# --- ROI-Auswahl ---
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    for p in points:
        cv2.circle(frame, p, 5, (0, 255, 0), -1)
    if len(points) > 1:
        cv2.polylines(frame, [np.array(points)], False, (0, 255, 0), 2)

    cv2.imshow("Kamera", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()
    if len(points) == 4:
        selecting = False
        print("✅ Vier Punkte gewählt – starte Farberkennung!")
        break

# --- Perspektivische Entzerrung vorbereiten ---
dst_size = 400  # Pixelgröße des entzerrten Bereichs (400×400 px = 18×18 cm)
cm_per_pixel = 18 / dst_size  # Maßstab in cm/px
stud_cm = 1.6  # Standard Lego-Noppe = 1.6 cm

src_pts = np.array(points, dtype="float32")
dst_pts = np.array([
    [0, 0],
    [dst_size-1, 0],
    [dst_size-1, dst_size-1],
    [0, dst_size-1]
], dtype="float32")

M = cv2.getPerspectiveTransform(src_pts, dst_pts)

# --- Hauptschleife: Farb- und Stein-Erkennung ---
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    warp = cv2.warpPerspective(frame, M, (dst_size, dst_size))
    hsv = cv2.cvtColor(warp, cv2.COLOR_BGR2HSV)

    for color, (lower, upper) in color_ranges.items():
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 300:  # kleine Rauschobjekte ignorieren
                continue

            x, y, w_box, h_box = cv2.boundingRect(cnt)
            cx, cy = x + w_box // 2, y + h_box // 2

            # --- Normalisierte Position (0–1 Koordinaten) ---
            nx = cx / dst_size
            ny = 1 - (cy / dst_size)  # 0 unten

            # --- Größe in cm berechnen ---
            brick_w_cm = w_box * cm_per_pixel
            brick_h_cm = h_box * cm_per_pixel

            # --- In Noppen umrechnen ---
            cols = max(1, round(brick_w_cm / stud_cm))
            rows = max(1, round(brick_h_cm / stud_cm))
            brick_type = f"{cols}x{rows}"

            # --- Anzeige ---
            cv2.rectangle(warp, (x, y), (x+w_box, y+h_box), (0, 255, 255), 2)
            cv2.putText(warp,
                        f"{color} {brick_type} ({nx:.2f},{ny:.2f})",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2)

    # --- Rahmen zeichnen ---
    cv2.rectangle(warp, (0, 0), (dst_size - 1, dst_size - 1), (100, 100, 100), 1)
    cv2.imshow("ROI", warp)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
