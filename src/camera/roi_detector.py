import cv2
import numpy as np

def select_roi(cap):
    selected_points = []
    selecting = True

    def mouse_click(event, x, y, flags, param):
        nonlocal selected_points, selecting
        if event == cv2.EVENT_LBUTTONDOWN and selecting:
            selected_points.append((x, y))
            print(f"Point {len(selected_points)}: {x},{y}")
            if len(selected_points) == 4:
                selecting = False

    cv2.namedWindow("Camera")
    cv2.setMouseCallback("Camera", mouse_click)
    print("Select 4 edges clockwise (start up left!)")

    while selecting:
        frame_available, frame = cap.read()
        if not frame_available:
            continue
        frame = cv2.flip(frame, 1)

        temp = frame.copy()
        for p in selected_points:
            cv2.circle(temp, p, 5, (0,255,0), -1)
        if len(selected_points) > 1:
            cv2.polylines(temp, [np.array(selected_points)], False, (0, 255, 0), 2)

        cv2.imshow("Camera", temp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            selecting = False
            break

    cv2.destroyWindow("Camera")
    return np.array(selected_points, dtype=np.float32)
