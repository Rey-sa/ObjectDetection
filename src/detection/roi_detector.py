import cv2
import numpy as np

def detect_roi_square(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.erode(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    for contour in contours[:5]:
        area = cv2.contourArea(contour)

        min_area = (frame.shape[0] * frame.shape[1]) * 0.05
        if area < min_area:
            continue

        permimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * permimeter, True)

        if len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)

            if 0.7 <= aspect_ratio < 1.3:
                points = approx.reshape(4, 2).astype(np.float32)
                points = sort_points_clockwise(points)

                return points
    return None

def sort_points_clockwise(points):
    """
    Sorts 4 Points clockwise, starting in the left upper corner.
    """
    points = points[points[:, 1].argsort()]

    top = points[:2]
    bottom = points[2:]

    top = top[top[:, 0].argsort()]
    bottom = bottom[bottom[:, 0].argsort()[::-1]]

    return np.array([top[0], top[1], bottom[0], bottom[1]], dtype=np.float32)


def select_roi(video, auto_detect=True):
    """
    Chooses ROI - automatic or manual on fallback.

    Args:
        video: VideoCapture Object
        auto_detect: If true, try automatic detection
    """
    selected_points = []
    selecting = True
    auto_detected = False

    def mouse_click(event, x, y, flags, param):
        nonlocal selected_points, selecting
        if event == cv2.EVENT_LBUTTONDOWN and selecting:
            selected_points.append((x, y))
            print(f"Point {len(selected_points)}: {x},{y}")
            if len(selected_points) == 4:
                selecting = False

    cv2.namedWindow("Camera")
    cv2.setMouseCallback("Camera", mouse_click)

    if auto_detect:
        print("Trying automatic roi detection...")
        for attempt in range(30):
            frame_available, frame = video.read()
            if not frame_available:
                continue
                # Fix orientation
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)


            detected = detect_roi_square(frame)

            if detected is not None:
                selected_points = [tuple(map(int, p)) for p in detected]
                auto_detected = True
                print(f"✓ Square ROI detected successfully!")
                print(f"Points: {selected_points}")

                temp = frame.copy()
                for i, p in enumerate(selected_points):
                    cv2.circle(temp, p, 5, (0, 255, 0), -1)
                    cv2.putText(temp, str(i + 1), (p[0] + 10, p[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.polylines(temp, [np.array(selected_points)], True, (0, 255, 0), 2)
                cv2.putText(temp, "Detected automatically! Closing window...",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Camera", temp)
                cv2.waitKey(2000)
                selecting = False
                break

            # Live stream while searching
            temp = frame.copy()
            cv2.putText(temp, f"Searching for square ROI... ({attempt + 1}/30)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Camera", temp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                selecting = False
                break

    # Fallback manual selection
    if not auto_detected and selecting:
        print("Automatic detection failed.")
        print("Manual selection: Click 4 edges clockwise starting in the upper left corner!")
        print("Press 'q' to exit.")

        selected_points = []
        selecting = True

        while selecting:
            frame_available, frame = video.read()
            if not frame_available:
                continue
                # Fix orientation
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            temp = frame.copy()
            for i, p in enumerate(selected_points):
                cv2.circle(temp, p, 5, (0, 255, 0), -1)
                cv2.putText(temp, str(i + 1), (p[0] + 10, p[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            if len(selected_points) > 1:
                cv2.polylines(temp, [np.array(selected_points)], False, (0, 255, 0), 2)

            cv2.putText(temp, f"Select Point {len(selected_points) + 1}/4",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow("Camera", temp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                selecting = False
                break

    cv2.destroyWindow("Camera")
    return np.array(selected_points, dtype=np.float32)