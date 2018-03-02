import os

import cv2
import numpy as np

import utils
from regionCutter import RegionCutter

pwd = os.getcwd()

img = cv2.imread(pwd + "/example_image.png", flags=cv2.IMREAD_COLOR)

# Initializes Region Cutter and sets its variables
region_cutter = RegionCutter()
region_cutter.set_img_shape(img.shape)
region_cutter.set_vertices()

cropped_img = region_cutter.cut_region(img)

cv2.imshow("cropped_image", cropped_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
