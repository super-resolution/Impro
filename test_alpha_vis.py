"""

"""
import time
import pyqtgraph as pg
import cv2
from processing.Image import microscope_image as image
from processing.Image import storm_image as storm
from processing.Filter import Filter
import numpy as np
from PyQt5.QtCore import QPoint
app = pg.mkQApp()
import pyqtgraph.opengl as gl
import pyqtgraph
import visualisation.Render as rend
import processing.CudaAlphaShape as alpha
from processing.CudaHough import hough_transform
from skimage import transform


i = image(r"D:\asdf\3D Auswertung 22092016\20160919_SIM1\20160919_SIM_0824a_lh_1_Out_Channel Alignment.czi")
i.parse()
y = i.data[:,7]
y= np.clip(y,0,255)
def create_alpha_shape():

    #i = image(r"D:\_3dSarahNeu\!!20170317\20170317_0308c_SIM10_Out_Channel Alignment.czi")
    data = storm(r"D:\asdf\3D Auswertung 22092016\20160919_SIM1\20190920_3D_sample0824a_SIM0919_SIM1.txt")
    data.parse()

    #todo: photon filter, z filter
    #x = Filter.photon_filter(data.stormData,3000,99999)
    x = Filter.local_density_filter(data.stormData, 100.0, 16)
    #x= data.stormData
    t1 = time.time()
    k_simplices = alpha.get_k_simplices(x[...,0:2])
    #x  = ksimplices[index[index]]
    print(str(time.time()-t1))

    #todo: fasten up
    t1 = time.time()
    points = np.empty((2*k_simplices.shape[0],5))
    points[::2,0] = x[(k_simplices[...,0]).astype(np.int32),0]
    points[::2,1] = x[(k_simplices[...,0]).astype(np.int32),1]
    points[1::2,0] = x[(k_simplices[...,1]).astype(np.int32),0]
    points[1::2,1] = x[(k_simplices[...,1]).astype(np.int32),1]
    points[...,2] = np.repeat(k_simplices[...,2],2,)
    points[...,3] = np.repeat(k_simplices[...,3],2,)
    points[...,4] = np.repeat(k_simplices[...,4],2,)

    #dat[...,4] = np.clip(dat[...,4],0,255)
    print("reshape points: " + str(time.time()-t1))
    #new = points
    x = gl.GLViewWidget()
    x.show()
    item2 = rend.alpha_complex(filename = r"\Alpha")
    item = rend.image2D(filename=r"\Image")

    item.set_data(y.astype(np.int32), [[1.0,1.0,1.0,1.0]], [0], smooth=True)
    #item2 = rend.raycast(filename = r"\Image")
    #item2.set_data(numpy.clip(i.data,0,255).astype(numpy.int32), [[],[1.0,1.0,1.0,1.0]], [1])
    x.addItem(item)
    #item.scale(1,1,0.05)
    item.show()
    item2.show()
    item2.set_data(position=points[...,0:2], simplices=points[...,2:5], alpha=130.0, size=float(20))
    item2.background_render(QPoint(1158,1411),1.0)
    im = np.array(item2.image)

#cv2.imshow("accum", im.astype("uint8"))

    cv2.imwrite(r"C:\Users\biophys\Desktop\Masterarbeit\src\abb\test_image.tif",im)
im = cv2.imread(r"C:\Users\biophys\Desktop\Masterarbeit\src\abb\test_image.tif",-1)
#im = cv2.blur(im, (4,4))
#cv2.imshow("im", im.astype("uint8"))
#im = cv2.Canny(im, 150,200)
#im = cv2.Canny(y.astype("uint8"), 150, 200)
y = (y[0]).astype("uint8")[1000:2400,200:1600]
y= np.flipud(np.fliplr(y))#todo:resize images
#y = y[465*2-50:465*2+50,630*2-50:630*2+50]
print(im.shape)

cv2.imshow("accum", im.astype("uint8"))
#cv2.waitKey(0)
#M = cv2.getRotationMatrix2D((im.shape[0]/2,im.shape[1]/2),15,1)
#im = cv2.warpAffine(im,M,(im.shape[0],im.shape[1]))

im2 = cv2.cvtColor(im, cv2.COLOR_RGBA2GRAY)
#y = np.fliplr(y[0].astype("uint8"))

H = hough_transform()
H.set_image(y)


t_list = []
z = cv2.cvtColor(y, cv2.COLOR_GRAY2RGB)
im = im.astype("uint8")
imgs_t = []
points1 = []
points2 = []
for i in range(5):
    for j in range(2):

#print(indices, maximum)
        k,j = 450+j*200,650+j*200
        points1.append(np.array((k+100,i*200+100)))
        im2new = im2[k:j,0+i*200:200+i*200]
        imnew = im[k:j,0+i*200:200+i*200]
        H.set_template(im2new)
        res = H.transform()
        t_list.append(res[0:2])
        points2.append(res[1][0:2])
        M = cv2.getRotationMatrix2D((imnew.shape[0]/2,imnew.shape[1]/2),res[0],1)
        imnew = cv2.warpAffine(imnew,M,(imnew.shape[0],imnew.shape[1]))
        imgs_t.append(imnew)

        for h in range(200):
            for k in range(200):
                #print(j-100+k,i-100+h)
                z[2*res[1][0]-100+h,2*res[1][1]-100+k] += imnew[h,k,0:3]
#cv2.imwrite(r"C:\Users\biophys\Desktop\Masterarbeit\src\abb\.jpg",z)
base = [0]
grid=[]
num = []
for i,ent1 in enumerate(t_list):
    row1 = int(i/2)
    col1 = i%2
    max=0
    print(i)
    for j,ent2 in enumerate(t_list):
        row = int(j/2)
        col = j%2
        val1 = ent1[1][0]-(col1-col)*100
        val2 = ent1[1][1]-(row1-row)*100
        if np.absolute(val1-ent2[1][0])<50 and np.absolute(val2-ent2[1][1])<50:
            print(True)
            max+=1
        else:
            print(False)
    if max>4:
        num.append(i)
        grid.append(ent1)
points1 = np.asarray(points1)
points2 = np.asarray(points2)
points1 = points1[np.asarray(num)]
points2 = points2[np.asarray(num)]
print(points1,points2)
p1 = np.float32(points1)
p1 = np.fliplr(p1)
p2 = np.float32(points2)*2
p2 = np.fliplr(p2)
M = transform.estimate_transform("affine",p1,p2)
dst = transform.warp(im,inverse_map=M.inverse)*255
dst = dst.astype(np.uint8)
cv2.imshow("res", dst)
#for i in range(dst.shape[0]-100):
#    for j in range(dst.shape[1]-100):
#        z[i,j] += dst[i,j][0:3]
#for i,res in enumerate(grid):
#        for h in range(200):
#            for k in range(200):
                #print(j-100+k,i-100+h)
#                z[2*res[1][0]-100+h,2*res[1][1]-100+k] += imgs_t[num[i]][h,k,0:3]
#print(grid)
#v = cv2.matchTemplate(y[0].astype("uint8"), im.astype("uint8"),2)

cv2.imshow("sdgag", z.astype(np.uint8))# img.astype("uint8"))
#cv2.waitKey(0)
cv2.imwrite(r"C:\Users\biophys\Desktop\Masterarbeit\src\abb\segments.jpg",z)
#app.exec()