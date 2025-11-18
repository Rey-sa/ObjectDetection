# Global configuration
CAMERA_INDEX = 1

WINDOW_NAME = "LIVE FEED"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
DESTINATION_SIZE = 400
FIELD_CM = 18.0
STUD_CM = 1.6

DEFAULT_MIN_THRESH = 150
DEFAULT_MAX_THRESH = 255
DEFAULT_MIN_AREA = 5000
DEFAULT_MAX_AREA = 30000

# ------------------------------
# HSV-Colorsections
# ------------------------------
COLOR_RANGES = {
    "red":   ([0, 120, 70], [10, 255, 255]),
    "red2":  ([170, 120, 70], [180, 255, 255]),
    "green": ([30, 50, 50], [85, 255, 200]),
    "blue":  ([90, 50, 50], [130, 255, 255]),
    "yellow":([20, 100, 100], [30, 255, 255]),
    "orange": ([10,120,100], [25,255,255])
}

# ------------------------------
# White-Masking (surpress reflexions)
# ------------------------------
WHITE_MASK_RANGE = {
    "lower": (0, 0, 180),
    "upper": (180, 60, 255)
}

