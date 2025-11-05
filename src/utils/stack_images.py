import cv2
import numpy as np


def stack_images(scale, vid_array):
    rows = len(vid_array)
    cols = len(vid_array[0])
    rows_available = isinstance(vid_array[0], list)
    width = vid_array[0][0].shape[1]
    height = vid_array[0][0].shape[0]
    if rows_available:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if vid_array[x][y].shape[:2] == vid_array[0][0].shape [:2]:
                    vid_array[x][y] = cv2.resize(vid_array[x][y], (0, 0), None, scale, scale)
                else:
                    vid_array[x][y] = cv2.resize(vid_array[x][y], (vid_array[0][0].shape[1], vid_array[0][0].shape[0]), None, scale, scale)
                if len(vid_array[x][y].shape) == 2: vid_array[x][y]= cv2.cvtColor(vid_array[x][y], cv2.COLOR_GRAY2BGR)
        image_blank = np.zeros((height, width, 3), np.uint8)
        hor = [image_blank]*rows
        hor_con = [image_blank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(vid_array[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if vid_array[x].shape[:2] == vid_array[0].shape[:2]:
                vid_array[x] = cv2.resize(vid_array[x], (0, 0), None, scale, scale)
            else:
                vid_array[x] = cv2.resize(vid_array[x], (vid_array[0].shape[1], vid_array[0].shape[0]), None, scale, scale)
            if len(vid_array[x].shape) == 2: vid_array[x] = cv2.cvtColor(vid_array[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(vid_array)
        ver = hor
    return ver

