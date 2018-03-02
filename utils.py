import os

import cv2
import numpy as np

def fillPoly(mask, vertices):
    """
    Wraps the opencv method of cv2.fillPoly
    """

    if len(mask.shape) > 2:
        channel_count = mask.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)
