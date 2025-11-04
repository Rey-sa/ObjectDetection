import cv2 as cv
import numpy as np

def stack_images(scale, vid_array):
    rows = len(vid_array)
    cols = len(vid_array[0]) if isinstance(vid_array[0], list) else len(vid_array)
    rows_available = isinstance(vid_array[0], list)

    width = vid_array[0][0].shape[1] if rows_available else vid_array[0].shape[1]
    height = vid_array[0][0].shape[0] if rows_available else vid_array[0].shape[0]

    if rows_available:
        for x in range(rows):
            for y in range(cols):
                vid = vid_array[x][y]

                # Resize Video
                if vid.shape[:2] != (height, width):
                    vid_array[x][y] = cv.resize(vid, (width, height), interpolation=cv.INTER_AREA)

                # Convert grayscale to BGR
                if len(vid_array[x][y].shape) == 2:
                    vid_array[x][y] = cv.cvtColor(vid_array[x][y], cv.COLOR_GRAY2BGR)

            # Horizontal / vertical chaining
            hor = [np.hstack(vid_array[x]) for x in range(rows)]
            ver = np.vstack(hor)
    else:
        for x in range(rows):
            if vid_array[x].shape[:2] != (height, width):
                vid_array[x] = cv.resize(vid_array[x], (width, height), interpolation=cv.INTER_AREA)

            if len(vid_array[x].shape) == 2:
                vid_array[x] = cv.cvtColor(vid_array[x], cv.COLOR_GRAY2BGR)

        ver = np.hstack(vid_array)

    # Scale result
    ver = cv.resize(ver, (0, 0), fx=scale, fy=scale)
    return ver