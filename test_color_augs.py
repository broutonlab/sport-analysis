import cv2
import numpy as np

# read the target file
image = "./data/raw/1_1m/3.jpg"
target_img = cv2.imread(image, 1)

print(target_img.shape)
# create an image with a single color (here: red)
red_img = np.full((target_img.shape[0], target_img.shape[1], 3), (0, 0, 255), np.uint8)

# add the filter  with a weight factor of 20% to the target image
fused_img = cv2.addWeighted(target_img, 0.8, red_img, 0.2, 0)

cv2.imshow("Red Filter", fused_img)

cv2.waitKey(0)

# cv2.imwrite('blagaj_red_filter.jpg', fused_img)
