import cv2
import numpy as np

from processing.CudaHough import hough_transform
from processing.Image import microscope_image as image
from processing.Image import storm_image as storm
from processing.Filter import Filter

#from libs import TestOpenCVHough

i = image(r"C:\Users\biophys\Desktop\test_20160602\sample2505_A3_vGlut647_bassoon532_Map2488_9_Out_Channel Alignment.czi")
i.parse()
im = np.clip(i.data[0,0],0,255).astype("uint8")
templ = cv2.resize(im[1000:1100, 1000:1100], (0,0), fx=1.0, fy=1.0)#consider rotation/scaling
#storm = storm(r"C:\Users\biophys\Desktop\test_20160602\20160602_hp_sample2505_A3_vGlut_Al647_3.txt")
#storm.parse()
#data = filter.frame_filter(storm.stormData, 300, 600)
#data = filter.local_density_filter(data, 100, 12)




#cv2.imshow('image',im[1500:1600,1500:1600])
H = hough_transform()
H.set_template(templ)
H.set_image(im[500:1500, 500:1500])
im = np.clip(H.transform(),0,255)
#im = cv2.blur(im, (3,3))
#im = cv2.Canny(im.astype("uint8"), 110, 140)
for i in range(500):
    for j in range(500):
        if im[i,j] == 255:
            print(i,j)

cv2.imshow("accum", im.astype("uint8"))
cv2.waitKey(0)
cv2.destroyAllWindows()
i,j = np.unravel_index(im.argmax(), im.shape)
print(i,j)


# edges = cv2.Canny(im, 150,200)

#
# x = TestOpenCVHough.general_hough_transform(im[1500:1600,1500:1600])
# im2 = im[1000:2000,1000:2000]
#
# y = x.findTemplate(im[1000:2000,1000:2000], int(30))
# m = 0
# #for j,i in enumerate(y):
#     #print(i.hits)
# #    if i.hits > m:
# #        m = j
# #        print(i.x())
# #        print(i.y())


