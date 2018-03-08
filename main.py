import os

import cv2

from utils.img_functions import abs_sobel_tresh
from utils.region_cutter import RegionCutter

pwd = os.getcwd()

img = cv2.imread(pwd + "/example_image.png", flags=cv2.IMREAD_COLOR)

# Initializes Region Cutter and sets its variables
region_cutter = RegionCutter()
region_cutter.set_img_shape(img.shape)
region_cutter.set_vertices()

img_cropped = region_cutter.cut_region(img)
img_cropped_gray = cv2.cvtColor(img_cropped, cv2.COLOR_RGB2GRAY)

gradx = abs_sobel_tresh(img_cropped_gray, orientation="x", kernel_size=7, thresh=(10, 255))
grady = abs_sobel_tresh(img_cropped_gray, orientation="y", kernel_size=7, thresh=(60, 255))

cv2.imshow("gradient in x", gradx[1])
cv2.imshow("gradient in y", grady[1])
cv2.waitKey(0)
cv2.destroyAllWindows()
