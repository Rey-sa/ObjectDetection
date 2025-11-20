import cv2
import numpy as np
from src.config.settings import DESTINATION_SIZE, FIELD_CM, STUD_CM

def get_brick_size_pca(cnt, cm_per_pixel, STUD_CM):

    # All contour points as float
    pts = cnt.reshape(-1, 2).astype(np.float32)

    # PCA
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(pts, mean=None)

    # Largest eigenvector = direction of length
    # eigenvalues = variance along axis → sqrt gives "spread"
    length  = 2 * np.sqrt(eigenvalues[0][0])
    width   = 2 * np.sqrt(eigenvalues[1][0])

    # Convert from pixels → cm
    length_cm = length * cm_per_pixel
    width_cm  = width * cm_per_pixel

    # Convert to stud count
    studs_long = max(1, round(length_cm / STUD_CM))
    studs_short = max(1, round(width_cm / STUD_CM))

    # Normalize so x ≥ y
    cols = max(studs_long, studs_short)
    rows = min(studs_long, studs_short)

    return cols, rows

def analyze_bricks(color, contours, warp):
    cm_per_pixel = FIELD_CM / DESTINATION_SIZE
    bricks = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue

        # smooth contour
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Detect L-shape (non-convex polygon)
        is_convex = cv2.isContourConvex(approx)

        x, y, w, h = cv2.boundingRect(cnt)

        # center in normalized coords
        cx, cy = x + w//2, y + h//2
        nx = cx / DESTINATION_SIZE
        ny = 1 - (cy / DESTINATION_SIZE)

        # convert to cm
        cols, rows = get_brick_size_pca(cnt, cm_per_pixel, STUD_CM)

        # final brick type
        if not is_convex:
            brick_type = "L"
        else:
            brick_type = f"{cols}x{rows}"

        bricks.append({
            "color": color,
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "nx": nx,
            "ny": ny,
            "type": brick_type
        })

    return bricks
