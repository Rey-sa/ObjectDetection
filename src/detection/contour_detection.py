import cv2 as cv
from src.detection.color_detection import detect_colors


def get_contours(img, output_img, original_frame, area_min):
    contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    frame_h, frame_w = original_frame.shape[:2]

    objects = []

    for contour in contours:
        area = cv.contourArea(contour)
        if area > area_min:
            x, y, w, h = cv.boundingRect(contour)

            # Crop roi
            roi = original_frame[y:y+h, x:x+w]

            # Recognize color
            color_name = detect_colors(roi)

            # Aspect Ratio
            aspect_ratio = w / h
            if aspect_ratio > 2.3:
                shape_type = "2x5"
            elif aspect_ratio > 1.5:
                shape_type = "2x1"
            elif aspect_ratio > 0.8:
                shape_type = "1x1"
            else:
                shape_type = "Unknown"

            # Normed position
            norm_x = x / frame_w
            norm_y = y / frame_h

            # Draw boxes + contours
            cv.drawContours(output_img, [contour], -1, (255, 0, 255), 2)
            cv.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv.putText(contour, f"{color_name} {shape_type} ({norm_x:.2f},{norm_y:.2f})",
                       (x, y - 25), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Save Object Infos
            objects.append({
                "contour": contour,
                "bbox": (x,y,w,h),
                "color": color_name,
                "shape": shape_type,
                "position": (norm_x, norm_y),
            })

    return objects