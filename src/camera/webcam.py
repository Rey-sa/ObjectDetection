import cv2 as cv

def init_camera(index,width, height):
    vid = cv.VideoCapture(index)
    vid.set(3, width)
    vid.set(4, height)
    return vid