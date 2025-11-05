import cv2 as cv
from .color_detection import detect_color

def get_contours(img, output_img, original_frame, area_min, window_name="Result"):
    contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    for contour in contours:
        area = cv.contourArea(contour)
        if area > area_min:
            perimeter = cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, 0.02 * perimeter, True)
            x, y, w, h = cv.boundingRect(approx)

            # Farberkennung aus Originalframe
            roi = original_frame[y:y+h, x:x+w]
            color_name = detect_color(roi)

            # Konturen und Boxen zeichnen
            cv.drawContours(output_img, [contour], -1, (255, 0, 255), 2)
            cv.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv.putText(output_img, f"{color_name}", (x, y - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv.putText(output_img, f"Points: {len(approx)}", (x + w + 20, y + 20),
                       cv.FONT_HERSHEY_COMPLEX, .7, (0, 255, 0), 2)
            cv.putText(output_img, f"Area: {int(area)}", (x + w + 20, y + 45),
                       cv.FONT_HERSHEY_COMPLEX, .7, (0, 255, 0), 2)
