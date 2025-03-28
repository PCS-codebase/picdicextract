import numpy as np
import cv2
from PIL import Image

def remove_background(pil_img):
    """
    General background removal using Otsu's thresholding.
    Converts the image to grayscale, applies a Gaussian blur, then uses Otsu's
    thresholding to generate a high-contrast binary image. This helps remove noise
    and various background colors.
    """
    # Convert to grayscale (if not already)
    gray = np.array(pil_img.convert("L"))
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply Otsu's thresholding
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # If the text is dark on a light background, invert the image if necessary
    if np.mean(thresh) > 127:
        thresh = cv2.bitwise_not(thresh)
    return Image.fromarray(thresh)