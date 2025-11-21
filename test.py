import cv2
import numpy as np


def nothing(x):
    pass


def detect_studs(image, min_radius=5, max_radius=30):
    """Erkennt kreisförmige Studs im Bild"""
    circles = cv2.HoughCircles(
        image,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=50,
        param2=30,
        minRadius=min_radius,
        maxRadius=max_radius
    )
    return circles


def classify_brick(stud_count, contour):
    """Klassifiziert Legosteine basierend auf Stud-Anzahl und Form"""
    if stud_count == 1:
        return "1x1"
    elif stud_count == 2:
        return "1x2"
    elif stud_count == 3:
        # Unterscheidung zwischen 1x3 und 2x3 über Bounding Box
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w, h) / min(w, h)
        if aspect_ratio > 2.5:
            return "1x3"
        else:
            return "2x3"
    elif stud_count == 4:
        return "2x2 oder 1x4"
    elif stud_count >= 5 and stud_count <= 8:
        # Unterscheidung 2x3 vs 2x4
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w, h) / min(w, h)
        if aspect_ratio > 1.7:
            return "2x4"
        else:
            return "2x3"
    else:
        return f"Unbekannt ({stud_count} Studs)"


video = cv2.VideoCapture(1)

# Fenster für Trackbars
cv2.namedWindow("LEGO Detection")
cv2.createTrackbar("Canny T1", "LEGO Detection", 10, 255, nothing)
cv2.createTrackbar("Canny T2", "LEGO Detection", 36, 255, nothing)
cv2.createTrackbar("Min Area", "LEGO Detection", 500, 5000, nothing)
cv2.createTrackbar("Stud Min R", "LEGO Detection", 5, 50, nothing)
cv2.createTrackbar("Stud Max R", "LEGO Detection", 25, 100, nothing)

while True:
    frame_available, frame = video.read()
    if not frame_available:
        continue

    # Fix orientation
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    output = frame.copy()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)

    # Lese Parameter von Trackbars
    t1 = cv2.getTrackbarPos("Canny T1", "LEGO Detection")
    t2 = cv2.getTrackbarPos("Canny T2", "LEGO Detection")
    min_area = cv2.getTrackbarPos("Min Area", "LEGO Detection")
    stud_min_r = cv2.getTrackbarPos("Stud Min R", "LEGO Detection")
    stud_max_r = cv2.getTrackbarPos("Stud Max R", "LEGO Detection")

    # Canny Edge Detection
    edges = cv2.Canny(blurred, t1, t2)

    # Morphologische Operationen um Konturen zu verbessern
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    # Finde Konturen
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Analysiere jede Kontur
    for contour in contours:
        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        # Zeichne Kontur
        cv2.drawContours(output, [contour], -1, (0, 255, 0), 2)

        # Erstelle Maske für diese Kontur
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        # Wende Maske an
        masked = cv2.bitwise_and(blurred, blurred, mask=mask)

        # Erkenne Studs in diesem Bereich
        circles = detect_studs(masked, stud_min_r, stud_max_r)

        stud_count = 0
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                cx, cy, r = circle
                # Prüfe ob Kreis innerhalb der Kontur liegt
                if cv2.pointPolygonTest(contour, (float(cx), float(cy)), False) >= 0:
                    cv2.circle(output, (cx, cy), r, (255, 0, 0), 2)
                    cv2.circle(output, (cx, cy), 2, (0, 0, 255), 3)
                    stud_count += 1

        # Klassifiziere den Stein
        brick_type = classify_brick(stud_count, contour)

        # Zeige Information
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(output, f"{brick_type}", (cx - 30, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(output, f"Studs: {stud_count}", (cx - 30, cy + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # Zeige Ergebnisse
    cv2.imshow("LEGO Detection", output)
    cv2.imshow("Edges", edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()