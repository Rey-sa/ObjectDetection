import cv2
import numpy as np
from typing import List, Optional, Union

def stack_images(scale: float,
                 vid_array: Union[List[np.ndarray], List[List[np.ndarray]]],
                 labels: Optional[Union[List[str], List[List[str]]]] = None) -> np.ndarray:
    """
    Combines images into a grid with scaling and grayscale-to-BGR conversion.
    Optionally displays text labels at the top-left of each image.

    Args:
        scale: Scaling factor
        vid_array: List of videos or list of lists of videos
        labels: Optional, same structure as vid_array with text labels

    Returns:
        stack: Stacked image (np.ndarray)
    """

    def process_image(vid: np.ndarray, label: str, width: int, height: int) -> np.ndarray:
        """Helper function to process a single image."""
        # Resize
        if vid.shape[:2] != (height, width):
            vid = cv2.resize(vid, (width, height), interpolation=cv2.INTER_AREA)
        vid = cv2.resize(vid, (0, 0), fx=scale, fy=scale)

        # Grayscale to BGR
        if len(vid.shape) == 2:
            vid = cv2.cvtColor(vid, cv2.COLOR_GRAY2BGR)

        # Add label
        if label:
            cv2.putText(vid, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_8)
        return vid

    rows = len(vid_array)
    rows_available = isinstance(vid_array[0], list)
    cols = len(vid_array[0]) if rows_available else len(vid_array)

    width = vid_array[0][0].shape[1] if rows_available else vid_array[0].shape[1]
    height = vid_array[0][0].shape[0] if rows_available else vid_array[0].shape[0]

    # Initialize labels
    if labels is None:
        if rows_available:
            labels = [[""] * cols for _ in range(rows)]
        else:
            labels = [""] * cols

    if rows_available:
        for x in range(rows):
            for y in range(cols):
                vid_array[x][y] = process_image(vid_array[x][y], str(labels[x][y]), width, height)

        # Horizontal / vertical adding
        hor = [np.hstack(vid_array[x]) for x in range(rows)]
        stack = np.vstack(hor)

    else:
        for x in range(rows):
            vid_array[x] = process_image(vid_array[x], str(labels[x]), width, height)

        stack = np.hstack(vid_array)

    return stack
