import cv2
import numpy as np
from src.config.settings import DESTINATION_SIZE

def compute_perspective_matrix(src_pts):
    dst_pts = np.array([
        [0, 0],
        [DESTINATION_SIZE-1, 0],
        [DESTINATION_SIZE-1, DESTINATION_SIZE-1],
        [0, DESTINATION_SIZE-1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return M

def warp_to_square(frame, M):
    return cv2.warpPerspective(frame, M, (DESTINATION_SIZE, DESTINATION_SIZE))
