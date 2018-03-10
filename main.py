import os
import sys

import cv2

from filter_pipeline import filter_pipeline

KERNEL_SIZE = 7

pwd = os.getcwd()
argv = sys.argv

try:
    img_filename = str(sys.argv[1])
except IndexError:
    raise IndexError("Image path must be provided")

img_path = pwd + "/" + img_filename

img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)

# Run the cv pipeline to get the cropped processed img
color_binary_cropped = filter_pipeline(img)
color_binary_cropped[color_binary_cropped == 1] = 255

cv2.imshow("color_binary_cropped", color_binary_cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()
