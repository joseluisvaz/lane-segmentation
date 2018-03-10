import os

import cv2

from utils.img_functions import component_sobel_tresh
from utils.img_functions import magnitude_sobel_thresh
from utils.img_functions import direction_sobel_thresh
from utils.region_cutter import RegionCutter

KERNEL_SIZE = 7


pwd = os.getcwd()

img = cv2.imread(pwd + "/example_image2.jpg", flags=cv2.IMREAD_COLOR)

img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img_hsl = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
s_channel = img_hsl[:, :, 2]

# Initializes Region Cutter and sets its variables
region_cutter = RegionCutter()
region_cutter.set_img_shape(img_gray.shape)
region_cutter.set_vertices()

grad_x = component_sobel_tresh(img_gray, orientation="x", kernel_size=KERNEL_SIZE, thresh=(10, 255))
grad_y = component_sobel_tresh(img_gray, orientation="y", kernel_size=KERNEL_SIZE, thresh=(60, 255))
mag_img = magnitude_sobel_thresh(img_gray, kernel_size=KERNEL_SIZE, thresh=(40, 255))
dir_img = direction_sobel_thresh(img_gray, kernel_size=KERNEL_SIZE, thresh=(.65, 1.05))

combined_components = cv2.bitwise_and(grad_x, grad_y)
combined_mag_dir = cv2.bitwise_and(mag_img, dir_img)

combined_binary = cv2.bitwise_or(combined_components, combined_mag_dir)

s_channel_binary = cv2.inRange(s_channel, 160, 255)
s_channel_binary[s_channel_binary == 255] = 1

color_binary = cv2.bitwise_or(s_channel_binary, combined_binary)

color_binary[color_binary == 1] = 255

img_cropped = region_cutter.cut_region(color_binary)

# Combined thresholding information
cv2.imshow("img_cropped", img_cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()
