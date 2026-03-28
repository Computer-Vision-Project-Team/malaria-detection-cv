import cv2
import numpy as np

# Gaussian Blur 
def apply_gaussian_blur(image):
    return cv2.GaussianBlur(image, (3, 3), 0)


# Segmentation for infected cells
def segment_parasite(image):
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