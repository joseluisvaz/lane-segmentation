import cv2
import numpy as np


def abs_sobel_tresh(img_grayscale, orientation="x", kernel_size=3, thresh=(0, 255)):

    if orientation == "x":
        img_sobel = cv2.Sobel(img_grayscale, cv2.CV_64F, 1, 0, ksize=kernel_size)
    elif orientation == "y":
        img_sobel = cv2.Sobel(img_grayscale, cv2.CV_64F, 0, 1, ksize=kernel_size)
    else:
        raise ValueError

    img_sobel = np.absolute(img_sobel)

    img_sobel_scaled = np.uint8(255*img_sobel/np.max(img_sobel))

    return cv2.threshold(img_sobel_scaled, thresh[0], thresh[1], cv2.THRESH_BINARY)


def fill_poly(mask, vertices):
    """
    Wraps the opencv method of cv2.fillPoly
    """

    if len(mask.shape) > 2:
        channel_count = mask.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)
