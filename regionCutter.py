import cv2
import numpy as np

import utils

class RegionCutter(object):

    def __init__(self):
        """
        Initializes Cutter with shape
        """
        self.shape = None
        self.vertices = None

    def set_img_shape(self, shape):
        self.shape = shape

    def set_vertices(self):
        if self.shape is None:
            raise ValueError

        left_bottom = (100, self.shape[0])
        right_bottom = (self.shape[1] - 100, self.shape[0])
        apex1 = (610, 160)
        apex2 = (680, 160)
        inner_left_bottom = (200, self.shape[0])
        inner_right_bottom = (1000, self.shape[0])
        inner_apex1 = (610,355)
        inner_apex2 = (680,355)
        self.vertices = np.array([[left_bottom, apex1, apex2, \
                                 right_bottom, inner_right_bottom, \
                                 inner_apex1, inner_apex2, inner_left_bottom]], 
                                 dtype=np.int32)
        
    def cut_region(self, img):
        if self.shape is None:
            raise ValueError

        if self.shape != img.shape:
            raise ValueError

        if self.vertices is None:
            raise ValueError
        
        mask = np.zeros_like(img)
        
        utils.fillPoly(mask, self.vertices)

        return cv2.bitwise_and(img, mask) 
