import os

import cv2
import numpy as np

import utils
from regionCutter import RegionCutter

pwd = os.getcwd()

img = cv2.imread(pwd + "/data/training/image_2/um_000066.png", 0)

# Initializes Region Cutter and sets its variables
region_cutter = RegionCutter()
region_cutter.set_img_shape(img.shape)
region_cutter.set_vertices()

cropped_img = region_cutter.cut_region(img)

cv2.imshow("cropped_image", cropped_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
