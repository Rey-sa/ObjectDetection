import cv2
from src.camera.webcam import init_camera
from src.camera.roi_detector import select_roi
from src.calibration.perspective import compute_perspective_matrix, warp_to_square
from src.detection.color_detection import detect_colors
from src.detection.brick_detection import analyze_bricks
from src.utils.visuals import stack_images
from src.config.settings import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT

# ------------------------------
# Start camera
# ------------------------------
video = init_camera(CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT)

for _ in range(10):
    frame_available, frame = video.read()
    if not frame_available:
        raise RuntimeError("No frame available")
    frame = cv2.flip(frame, 1)

# ------------------------------
# ROI-Selection (live)
# ------------------------------
source_points = select_roi(video)

# ------------------------------
# Calculate perspective matrix
# ------------------------------
perspective_matrix = compute_perspective_matrix(source_points)

# ------------------------------
# Start main loop
# ------------------------------
while True:
    frame_available, frame = video.read()
    if not frame_available:
        continue
    frame = cv2.flip(frame, 1)

    warp = warp_to_square(frame, perspective_matrix)
    if warp is None:
        continue

    hsv = cv2.cvtColor(warp, cv2.COLOR_BGR2HSV)

    # Color & shape detection
    for color, contours in detect_colors(hsv):
        analyze_bricks(color, contours, warp)

    # Show stacked result
    stacked = stack_images(
        scale=0.7,
        vid_array=[[frame, warp]],
        labels=[["Original", "ROI"]]
    )
    cv2.imshow("Result", stacked)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ------------------------------
# Clean up
# ------------------------------
video.release()
cv2.destroyAllWindows()
