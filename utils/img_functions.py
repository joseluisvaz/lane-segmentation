import cv2
import numpy as np


def component_sobel_tresh(img_grayscale, orientation="x", kernel_size=3, thresh=(0, 255)):

    if orientation == "x":
        img_sobel = cv2.Sobel(img_grayscale, cv2.CV_64F, 1, 0, ksize=kernel_size)
    elif orientation == "y":
        img_sobel = cv2.Sobel(img_grayscale, cv2.CV_64F, 0, 1, ksize=kernel_size)
    else:
        raise ValueError

    img_sobel = np.absolute(img_sobel)
    img_sobel_scaled = np.uint8(255*img_sobel/np.max(img_sobel))

    return cv2.threshold(img_sobel_scaled, thresh[0], thresh[1], cv2.THRESH_BINARY)


def magnitude_sobel_thresh(img_grayscale, kernel_size=3, thresh=(0,255)):

    grad_x = cv2.Sobel(img_grayscale, cv2.CV_64F, 1, 0, ksize=kernel_size)
    grad_y = cv2.Sobel(img_grayscale, cv2.CV_64F, 0, 1, ksize=kernel_size)

    gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
    gradient_mag_scaled = np.uint8(255*gradient_mag / np.max(gradient_mag))

    return cv2.threshold(gradient_mag_scaled, thresh[0], thresh[1], cv2.THRESH_BINARY)


def direction_sobel_thresh(img_grayscale, kernel_size=3, thresh=(0, np.pi/2)):

    grad_x = cv2.Sobel(img_grayscale, cv2.CV_64F, 1, 0, ksize=kernel_size)
    grad_y = cv2.Sobel(img_grayscale, cv2.CV_64F, 0, 1, ksize=kernel_size)

    # To ignore division and invalid errors

    with np.errstate(divide="ignore", invalid="ignore"):
        gradient_abs_direction = np.absolute(np.arctan(grad_y/grad_x))
        return cv2.threshold(gradient_abs_direction, thresh[0], thresh[1], cv2.THRESH_BINARY)


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
