import cv2 as cv
import numpy as np

# Calculate canny thresholds based on median value.
def auto_canny_thresholds(vid_gray):
    v = np.median(vid_gray)
    min_val = int((max(0, 0.66 * v)))
    max_val = int((min(255, 1.33 * v)))
    return min_val, max_val