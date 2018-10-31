"""

"""
import time
import pyqtgraph as pg
import cv2
import numpy as np

from .processing.Image import microscope_image as image
from .processing.Image import storm_image as storm
from .processing.Filter import filter
from .processing.utility import *


i = image(r"D:\asdf\3D Auswertung 22092016\20160919_SIM1\20160919_SIM_0824a_lh_1_Out_Channel Alignment.czi")
#i = image(r"D:\_3dSarahNeu\!!20170317\20170317_0308c_SIM9_Out_Channel Alignment.czi")
#i = image(r"D:\Microtuboli\Image2b_Al532_ex488_Structured Illumination.czi")
i.parse()
y = i.data[:,7]
#y = i.data[:,0]/2
y= np.clip(y[0],0,255)

# i = image(r"D:\_3dSarahNeu\!!20170317\20170317_0308c_SIM10_Out_Channel Alignment.czi")
data = storm(r"D:\asdf\3D Auswertung 22092016\20160919_SIM1\20190920_3D_sample0824a_SIM0919_SIM1.txt")
#data = storm(r"D:\_3dSarahNeu\!!20170317\trans_20170317_0308c_SIM9_RG_300_300_Z_coordinates_2016_11_23.txt")
#data = storm(r"D:\Microtuboli\20151203_sample2_Al532_Tub_1.txt")

data.parse()

# todo: photon filter, z filter
# x = Filter.photon_filter(data.stormData,3000,99999)
storm_data = filter.local_density_filter(data.stormData, 100.0, 18)


im = create_alpha_shape(storm_data)
storm_image = create_storm(storm_data)
#cv2.imwrite(r"C:\Users\biophys\Desktop\Masterarbeit\src\abb\test\storm.jpg",storm_image)
#cv2.imshow("asdf",storm_image.astype("uint8"))
#cv2.waitKey(0)
#buffer image
cv2.imwrite(r"C:\Users\biophys\Desktop\Masterarbeit\src\abb\test\alpha2.jpg",im)

#im = cv2.imread(r"C:\Users\biophys\Desktop\Masterarbeit\src\abb\test_image.tif",-1)

#check alpha shape


y = (y).astype("uint8")[1000:2400,200:1600]
y= np.flipud(np.fliplr(y))#todo:resize images
#y =np.flipud(y).astype(np.uint8)
#y = (y).astype("uint8")[0:1700,0:1700]

#cv2.imwrite(r"C:\Users\biophys\Desktop\Masterarbeit\src\abb\test\SIM2.jpg",y)
#cv2.imshow("microSIM", y)
#cv2.waitKey(0)
col = int(im.shape[1]/200)
row = int(im.shape[0]/200)
#cv2.imwrite(r"C:\Users\biophys\Desktop\Masterarbeit\src\abb\weighting\image.png",im.astype(np.uint8))
#cv2.waitKey(0)
norm_source = np.linalg.norm(cv2.cvtColor(im,cv2.COLOR_RGBA2GRAY))
norm_target = np.linalg.norm(y)
y = (y.astype(np.float32) * (norm_source/norm_target)).astype(np.uint8)
points1,points2,z,t_list = find_mapping(y, im,n_row=row,n_col=col)
print(z.max())
z[z>255] = 0
#cv2.imwrite(r"C:\Users\biophys\Desktop\Masterarbeit\src\abb\weighting\not_weighted.png",z.astype(np.uint8))
cv2.imshow("sdgag", z.astype(np.uint8))

#cv2.waitKey(0)
#cv2.imwrite(r"C:\Users\biophys\Desktop\Masterarbeit\src\abb\Hough_complete_.jpg",z)


p1, p2 = error_management(t_list, points1, points2,n_row = row)

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

mask = np.zeros_like(y)
mask[0:im.shape[0], 0:im.shape[1]] = 1

asdf = np.zeros_like(y)
asdf[0:im.shape[0], 0:im.shape[1]] = cv2.cvtColor(im, cv2.COLOR_RGBA2GRAY)

test_pearson(y, asdf ,mask , M)


color_warp = np.zeros_like(pearsonRGB)
color_warp[0:im.shape[0], 0:im.shape[1]] = cv2.cvtColor(im, cv2.COLOR_RGBA2RGB)
dst = transform.warp(color_warp,inverse_map=M.inverse)*255
dst = dst
cv2.imshow("res", dst)
added = dst.astype(np.uint16) + pearsonRGB.astype(np.uint16)
added[added>255] = 0
#cv2.imwrite(r"C:\Users\biophys\Desktop\Masterarbeit\src\abb\microtub.png",added.astype(np.uint8))
cv2.imshow("added", added.astype(np.uint8))


cv2.imshow("sdgag", z.astype(np.uint8))# img.astype("uint8"))
cv2.waitKey(0)

#test data1
#Pearson :0.324,ld100;18;alpha130
#test data2
#slice 0; ld100;5;alpha 100
#test data3
#slice 4; ld100; 18; alpha 130

#4541111115632106