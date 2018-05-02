import pyqtgraph as pg
import cv2
from processing.Image import microscope_image as image
from processing.Image import storm_image as storm
import numpy as np
from PyQt5.QtCore import QPoint
app = pg.mkQApp()
import pyqtgraph.opengl as gl
import visualisation.Render as rend
import test_alpha as alph
import numpy
i = image(r"C:\Users\biophys\Desktop\test_20160602\sample2505_A3_vGlut647_bassoon532_Map2488_9_Out_Channel Alignment.czi")
#i = image(r"D:\_3dSarahNeu\!!20170317\20170317_0308c_SIM10_Out_Channel Alignment.czi")
data = storm(r"C:\Users\biophys\Desktop\test_20160602\20160602_hp_sample2505_A3_vGlut_Al647_3.txt")
data.parse()
x= data.stormData[...,0:2]

dat = data.stormData
a = alph.alpha_complex(x)
li = []
for simp in a.k_simplices:
    simp.set_alpha(100)
    if simp.surface or simp.line:
        li.append(simp)

dat = []
for i in li:
    dat.append(a.points[i.indices][0])
    dat.append(a.points[i.indices][1])
#dat[...,4] = np.clip(dat[...,4],0,255)
#i.parse()
x = gl.GLViewWidget()
x.show()
item = rend.points(filename = r"\STORM2")
#y = i.data[:,0]/5
#item.set_data(numpy.clip(y,0,255).astype(numpy.int32), [[1.0,1.0,1.0,1.0]], [0], smooth=True)
#item2 = rend.raycast(filename = r"\Image")
#item2.set_data(numpy.clip(i.data,0,255).astype(numpy.int32), [[],[1.0,1.0,1.0,1.0]], [1])
dat = np.array(dat)
new = np.zeros((dat.shape[0],5))
new[...,0:2] = dat[...,0:2]
new[...,4] = 256
x.addItem(item)
item.scale(1,1,0.05)
item.show()
item.set_data(position=new, size=float(20), maxEmission=256.0)
#item.background_render(QPoint(1000,1000),1.0)
#im = np.array(item.image)
#im = cv2.Canny(im.astype("uint8"), 150, 200)
#cv2.imshow("accum", im.astype("uint8"))
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#item.show()
app.exec()