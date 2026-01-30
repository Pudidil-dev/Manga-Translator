"""Process speech bubbles - make bubble contents white."""

from typing import Tuple
import cv2
import numpy as np


def process_bubble(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process the speech bubble in the given image, making its contents white.

    Args:
        image: Input BGR image.

    Returns:
        Tuple of (processed image, largest contour).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest_contour], -1, 255, cv2.FILLED)

    image[mask == 255] = (255, 255, 255)

    return image, largest_contour
