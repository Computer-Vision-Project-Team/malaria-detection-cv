"""Image processing utilities for malaria cell classification.

Contains the Gaussian blur preprocessing function used by ImageDataGenerator
(Method 1: Image Enhancement) and parasite segmentation for visualization.
"""

import cv2
import numpy as np


def apply_gaussian_blur(img):
    """Apply a 3x3 Gaussian blur to preserve sharp parasite edges."""
    blurred_img = cv2.GaussianBlur(img, (3, 3), 0)
    return blurred_img


def segment_parasite(image):
    """Segment infected cells by detecting purple parasite regions."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Purple parasite color range
    lower = np.array([120, 50, 50])
    upper = np.array([170, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    # Clean mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = image.copy()

    for cnt in contours:
        if cv2.contourArea(cnt) > 50:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return output
