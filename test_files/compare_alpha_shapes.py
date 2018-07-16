import cgalpy
import time
import processing.CudaAlphaShape as alpha
from processing.Image import storm_image as storm
from processing.Filter import Filter
import numpy as np
import processing.CudaAlphaShape as alphaCuda
import cv2
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import visualisation.Render as rend
from PyQt5.QtCore import QPoint
app = pg.mkQApp()

data = storm(r"D:\asdf\3D Auswertung 22092016\20160919_SIM1\20190920_3D_sample0824a_SIM0919_SIM1.txt")
data.parse()
x= []
#todo: create small random data set
for i in range(1000):
    x.append(np.array([np.random.ranf(),np.random.ranf()])*10000)
x = np.asarray(x)
#x = Filter.photon_filter(data.stormData,3000,99999)
x = Filter.local_density_filter(data.stormData, 100, 16)


CGAL_point_list =[]
for i in x:
    CGAL_point_list.append(cgalpy.Point_2(i[0].astype(np.float64),i[1].astype(np.float64)))
calpha = cgalpy.Alpha_shape_2(np.int32(16900),CGAL_point_list)
edges = calpha.get_alpha_segments(calpha.Classification_type.REGULAR)
#edges += calpha.get_alpha_segments(calpha.Classification_type.SINGULAR)

pointsd= []
for i in edges:
    pointsd.append(np.asarray((i.x(),i.y(), 0, 50, 150)))
pointsd = np.asarray(pointsd)



k_simplices = alphaCuda.get_k_simplices(x[...,0:2])[0]

points = np.empty((2*k_simplices.shape[0],5))
points[::2,0] = x[(k_simplices[...,0]).astype(np.int32),0]
points[::2,1] = x[(k_simplices[...,0]).astype(np.int32),1]
points[1::2,0] = x[(k_simplices[...,1]).astype(np.int32),0]
points[1::2,1] = x[(k_simplices[...,1]).astype(np.int32),1]
points[...,2] = np.repeat(k_simplices[...,2],2,)
points[...,3] = np.repeat(k_simplices[...,3],2,)
points[...,4] = np.repeat(k_simplices[...,4],2,)

def test_point_consistency(cuda_points, cgal_points, alpha):
    delete = []
    for j,i in enumerate(cuda_points):
        if i[3] < alpha and i[4] > alpha:
            delete.append(j)
    points_alpha = cuda_points[delete]
    points_unique = np.unique(points_alpha[...,0:2],return_index=False, axis=0)
    points_cgal_unique = np.unique(cgal_points[...,0:2],return_index=False, axis=0)
    match_number = 0
    for i in points_unique:
       for j in points_cgal_unique:
           if i[0]==j[0] and i[1]==j[1]:
               match_number +=1
    if match_number == len(points_unique):
        print("Same Point data")

x = gl.GLViewWidget()
x.show()
item = rend.alpha_complex(filename = r"\Alpha")
x.addItem(item)
item.show()
item.set_data(position=points[...,0:2], simplices=points[...,2:5], alpha=130.0, size=float(2))
item.background_render(QPoint(1158,1411),4.0, rect=(0, 0, 200, 200))
im = np.array(item.image)
cv2.imwrite(r"C:\Users\biophys\Desktop\Masterarbeit\src\abb\Alpha_comparison_130_mine.jpeg",im)
app.exec()