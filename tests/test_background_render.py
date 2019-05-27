from impro.render.render import points
from PyQt5.QtCore import QPoint
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import numpy as np

app = pg.mkQApp()


"""functional test rendering points"""

storm_data = np.zeros((6,5))
storm_data[...,4] = 256
storm_data[...,0] = np.array([10000,10000,0,5000,7000,6000])
storm_data[...,1] = np.array([20000,0,0,5000,7000,6000])

widget = gl.GLViewWidget()
widget.show()
sizex = storm_data[..., 0].max() / 10
sizey = storm_data[..., 1].max() / 10
item2 = points(filename=r"\STORM2")
#item2.show()
size = 1000
item2.set_data(position=storm_data, size=float(size), maxEmission=256.0, color=np.array([1.0, 0.0, 0.0, 1.0]),
               )
item2.background_render(QPoint(sizex, sizey), 1.0)
im = np.array(item2.image)
for i in range(storm_data.shape[1]):
    point = storm_data[i,(1,0)]/10-1
    point = point.astype(np.int32)
    if im[point[0],point[1],0] == 0:
        raise ValueError("There should be a point on 2000,1000,0")
print("All points are rendered at the correct position. TEST PASSED")