"""

"""
import time
import pyqtgraph as pg
import cv2
import numpy as np

from processing.Image import microscope_image as image
from processing.Image import storm_image as storm
from processing.Filter import Filter
from processing.utility import *


#i = image(r"D:\asdf\3D Auswertung 22092016\20160919_SIM1\20160919_SIM_0824a_lh_1_Out_Channel Alignment.czi")
i = image(r"D:\_3dSarahNeu\!!20170317\20170317_0308c_SIM9_Out_Channel Alignment.czi")
i.parse()
#y = i.data[:,7]
y = i.data[:,1]/2
y= np.clip(y[0],0,255)

# i = image(r"D:\_3dSarahNeu\!!20170317\20170317_0308c_SIM10_Out_Channel Alignment.czi")
# data = storm(r"D:\asdf\3D Auswertung 22092016\20160919_SIM1\20190920_3D_sample0824a_SIM0919_SIM1.txt")
data = storm(r"D:\_3dSarahNeu\!!20170317\trans_20170317_0308c_SIM9_RG_300_300_Z_coordinates_2016_11_23.txt")
data.parse()

# todo: photon filter, z filter
# x = Filter.photon_filter(data.stormData,3000,99999)
storm_data = Filter.local_density_filter(data.stormData, 100.0, 8)

im = create_alpha_shape(storm_data)

#buffer image
#cv2.imwrite(r"C:\Users\biophys\Desktop\Masterarbeit\src\abb\test_image.tif",im)
#im = cv2.imread(r"C:\Users\biophys\Desktop\Masterarbeit\src\abb\test_image.tif",-1)

#check alpha shape
#cv2.imshow("im", im.astype("uint8"))


#y = (y[0]).astype("uint8")[1000:2400,200:1600]
y= np.flipud(np.fliplr(y))#todo:resize images
y = (y).astype("uint8")[0:1700,0:1700]


#cv2.imshow("microSIM", y)
#cv2.imshow("microdStorm", im)
#cv2.waitKey(0)
print(im.shape)

cv2.imshow("accum", im.astype("uint8"))

points1,points2,z,t_list = find_mapping(y, im)
print(z.max())
z = (0.5*z).astype(np.uint8)
cv2.imshow("sdgag", z.astype(np.uint8))
#cv2.waitKey(0)
#cv2.imwrite(r"C:\Users\biophys\Desktop\Masterarbeit\src\abb\Hough_complete_.jpg",z)


p1, p2 = error_management(t_list, points1, points2)

M = transform.estimate_transform("affine",p1,p2)
#z = z[200:1600,1000:2400]
# landmarks = np.array([[[87	,1311]
#     ,[566,865]
#     ,[221,782]
#     ,[389,1284]]
#     ,[[87,1311]
#     ,[516,793]
#     ,[158,750]
#     ,[391,1244]]])
# M = transform.estimate_transform("affine", landmarks[0], landmarks[1])
#test.test_pearson(z[100:1400,0:1100], im[0:1300, 0:1100, 0:3] , M)
pearsonRGB = cv2.cvtColor(y, cv2.COLOR_GRAY2RGB)
test_pearson(y[0:1000,0:1000], cv2.cvtColor(im[0:1000, 0:1000],cv2.COLOR_RGBA2GRAY) , M)

dst = transform.warp(im,inverse_map=M.inverse)*255
dst = dst.astype(np.uint8)
cv2.imshow("res", dst)
added = (dst[0:1000,0:1000,0:3] + pearsonRGB[0:1000,0:1000].astype(np.uint8))
#cv2.imwrite(r"C:\Users\biophys\Desktop\Masterarbeit\src\abb\micro_tub_aligned.png",added.astype(np.uint8))
cv2.imshow("added", added.astype(np.uint8))


cv2.imshow("sdgag", z.astype(np.uint8))# img.astype("uint8"))
cv2.waitKey(0)
