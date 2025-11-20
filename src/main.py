import cv2
from src.camera.webcam import init_camera
from src.detection.roi_detector import select_roi
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
    # Fix orientation
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

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
    # Fix orientation
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    warp = warp_to_square(frame, perspective_matrix)
    if warp is None:
        continue

    hsv = cv2.cvtColor(warp, cv2.COLOR_BGR2HSV)



    # Smoothing
    if 'prev_bricks' not in locals():
        prev_bricks = []
    smooth_factor = 0.7
    current_bricks = []

    # Color & shape detection
    for color, contours in detect_colors(hsv):
        bricks = analyze_bricks(color, contours, warp)
        current_bricks.extend(bricks)

    # Position smoothing
    smoothed_bricks = []
    for i, brick in enumerate(current_bricks):
        if i < len(prev_bricks) and prev_bricks[i]["color"] == brick["color"]:
            p = prev_bricks[i]
            for k in ["x", "y", "width", "height"]:
                brick[k] = int(p[k] * smooth_factor + brick[k] * (1 - smooth_factor))
        smoothed_bricks.append(brick)

    prev_bricks = smoothed_bricks

    for brick in smoothed_bricks:
        x,y,w,h = brick["x"], brick["y"], brick["width"], brick["height"]
        color = brick["color"]
        label = f"{color} {brick['type']} ({brick['nx']:.2f},{brick['ny']:.2f})"

        cv2.rectangle(warp, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(warp, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
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
