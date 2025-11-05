import cv2 as cv
import numpy as np
from utils.stack_images import stack_images
from utils.contour_detection import get_contours
from camera.webcam import init_camera
from utils.color_detection import detect_color
from config import *

def callback(x):
    pass

video = init_camera(CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT)
cv.namedWindow(WINDOW_NAME)
cv.resizeWindow(WINDOW_NAME, FRAME_WIDTH, FRAME_HEIGHT)
cv.createTrackbar('Min_Thresh', WINDOW_NAME, DEFAULT_MIN_THRESH, 255, callback)
cv.createTrackbar('Max_Thresh', WINDOW_NAME, DEFAULT_MAX_THRESH, 255, callback)
cv.createTrackbar('Area', WINDOW_NAME, DEFAULT_MIN_AREA, DEFAULT_MAX_AREA, callback)

while True:
    frame_available, vid = video.read()
    if not frame_available:
        break

    vid_contours = vid.copy()
    vid_blur = cv.GaussianBlur(vid, (7,7), 1)
    vid_gray = cv.cvtColor(vid_blur, cv.COLOR_BGR2GRAY)

    min_thresh = cv.getTrackbarPos('Min_Thresh', WINDOW_NAME)
    max_thresh = cv.getTrackbarPos('Max_Thresh', WINDOW_NAME)
    min_area = cv.getTrackbarPos('Area', WINDOW_NAME)

    vid_canny = cv.Canny(vid_gray, min_thresh, max_thresh)
    kernel = np.ones((5,5))
    vid_dilated = cv.dilate(vid_canny, kernel, iterations=1)

    get_contours(vid_dilated, vid_contours, min_area, WINDOW_NAME)
    color_detected = detect_color(vid)
    cv.putText(vid_contours, f"Color: {color_detected}", (30, 50), cv.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),2)

    combined = stack_images(
        0.8,
        [
            [vid, vid_blur, vid_gray],
            [vid_canny, vid_dilated, vid_contours]
        ],
        labels=[
            ["Original", "Blur", "Gray"],
            ["Canny", "Dilated", "Contours"]
        ]
    )

    cv.imshow(WINDOW_NAME, combined)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv.destroyAllWindows()
