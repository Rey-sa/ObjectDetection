import cv2
import numpy as np
from typing import List, Optional, Union

def resize_to_same_height(images: list, scale: float):
    # Bestimme die maximale Höhe nach Skalierung
    heights = [int(img.shape[0] * scale) for img in images]
    max_height = max(heights)

    resized = []
    for img in images:
        h, w = img.shape[:2]
        new_w = int(w * max_height / h)  # Breite proportional
        resized_img = cv2.resize(img, (new_w, max_height), interpolation=cv2.INTER_AREA)
        resized.append(resized_img)
    return resized



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

    def process_image(vid: np.ndarray, label: str) -> np.ndarray:
        """Helper function to process a single image."""
        # Resize proportional
        vid = cv2.resize(vid, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        # Grayscale → BGR
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

    # Initialize labels
    if labels is None:
        if rows_available:
            labels = [[""] * cols for _ in range(rows)]
        else:
            labels = [""] * cols

    if rows_available:
        hor = []
        for x in range(rows):
            row_imgs = [process_image(img, str(labels[x][y])) for y, img in enumerate(vid_array[x])]
            row_imgs = resize_to_same_height(row_imgs, 1.0)  # Skalierung schon in process_image
            hor.append(np.hstack(row_imgs))
        stack = np.vstack(hor)

    else:
        for x in range(rows):
            vid_array[x] = process_image(vid_array[x], str(labels[x]))

        stack = np.hstack(vid_array)

    return stack
