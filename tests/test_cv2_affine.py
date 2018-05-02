import numpy as np
import cv2
from skimage import transform

im = cv2.imread(r"C:\Users\biophys\Desktop\Masterarbeit\src\abb\test_image.tif",-1)

points1 = [[550, 100],
 [550, 300],
 [750, 300],
 [550, 500],
 [750, 500],
 [750, 700],
 [550, 900],
 [750, 900]]
points2 = [[152, 145],
 [138, 238],
 [232, 249],
 [125, 330],
 [220, 342],
 [208, 436],
 [104, 515],
 [198, 528]]
p1 = np.float32(points1)
p1 = np.fliplr(p1)
p2 = np.float32(points2)*2
p2 = np.fliplr(p2)
M = transform.estimate_transform("affine",p1,p2)
dst = transform.warp(im,inverse_map=M.inverse)*255
#M = cv2.getPerspectiveTransform(p1,p2)
#dst = cv2.warpAffine(im,M,(im.shape[0],im.shape[1]))
cv2.imshow("res", dst.astype(np.uint8))
cv2.waitKey(0)