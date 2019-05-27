import pyqtgraph as pg
from impro.data.image import MicroscopeImage as image
import numpy as np

app = pg.mkQApp()
import pyqtgraph.opengl as gl
from impro.render import render as rend

i = image(r"D:\Microtuboli\Image2b_Al532_ex488_Structured Illumination.czi")
#i = image(r"D:\_3dSarahNeu\!!20170317\20170317_0308c_SIM10_Out_Channel Alignment.czi")

i.parse()
from OpenGL.GL import *
from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem


class random_plot(GLGraphicsItem):
    def __init__(self):
        super().__init__()
        self.data = np.random.randint(100, size=(1,3)).astype(np.float32)
        self.data_heatmap = np.zeros((10,10))

    def set_data(self, data):
        self.data = data

    def paint(self):
        self.draw_grid()
        #self.draw_line_plot()
        #self.draw_scatter_plot()
        self.draw_heat_map()


    def draw_line_segment(self, start, end, width):
        glLineWidth(width)
        glBegin(GL_LINES)
        glColor4f(1.0,1.0,1.0,1.0)
        glVertex3f(start[0], start[1], start[2])
        glVertex3f(end[0], end[1], end[2])
        glEnd()

    def draw_point(self, size, pos):
        glPointSize(size)
        glBegin(GL_POINTS)
        glColor4f(1.0,0.0,1.0,1.0)
        glVertex3f(pos[0],pos[1],pos[2])
        glEnd()

    def draw_scatter_plot(self):
        glEnable(GL_POINT_SMOOTH)
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        for i in self.data:
            self.draw_point(10, i)

    def draw_line_plot(self):
        for i in range(len(self.data)-1):
            start = self.data[i]
            end = self.data[i+1]
            self.draw_line_segment(start,end, 0.5)

    def draw_grid(self, width=50):
        n = int(1000/(2*width))*2
        for i in range(n):
            j = i-n/2
            v1 = np.array([-500,j*width,0], dtype=np.float32)
            v2 = np.array([500,j*width,0], dtype=np.float32)
            self.draw_line_segment(v1,v2,3)
            h1 = np.array([j*width,-500,0], dtype=np.float32)
            h2 = np.array([j*width,500,0], dtype=np.float32)
            self.draw_line_segment(h1,h2,3)

    def draw_heat_map(self):
        d_max = self.data_heatmap.max()
        d_min = self.data_heatmap.min()
        half = (d_max+d_min)/2
        glPointSize(3)
        glBegin(GL_POINTS)
        for i in range(self.data_heatmap.shape[0]):
            for j in range(self.data_heatmap.shape[1]):
                value = self.data_heatmap[i,j]
                b = 1.0 - value / half
                r = value / half - 1.0
                if b<0:
                    b=0
                if r<0:
                    r=0
                g = 1.0 - b - r
                glColor4f(r, g, b, 0.5)
                glVertex3f(i-500,j-500,0)
        glEnd()


x = gl.GLViewWidget()
x.show()
item = rend.volume_rendering(filename = r"\volume\volume")
#y = i.data[:,0]/5
#item.set_data(numpy.clip(y,0,255).astype(numpy.int32), [[1.0,1.0,1.0,1.0]], [0], smooth=True)
#item2 = rend.raycast(filename = r"\Image")
#item2.set_data(numpy.clip(i.data,0,255).astype(numpy.int32), [[],[1.0,1.0,1.0,1.0]], [1])
z = random_plot()
x.addItem(z)
z.show()
def draw_sin():
    a = np.linspace(-np.pi, np.pi, 201)
    b = np.sin(a)

    data = np.zeros((201,3), dtype=np.float32)
    for i in range(len(a)):
        data[i,0] = a[i]
        data[i,1] = b[i]
    z.set_data(data)

def draw_heatmap():
    data = np.zeros((1000,1000),dtype=np.float32)
    sigma = 300
    for i in range(1000):
        i -= 500
        for j in range(1000):
            j -= 500
            t = np.exp(-0.5*(i**2)/(sigma**2)-0.5*(j**2)/(sigma**2))/(sigma**2*2*np.pi)
            data[i,j] = t
    z.data_heatmap = data
draw_heatmap()
item.scale(1,1,0.05)
item.show()
#item.set_data(data=i.data*10)
#item.background_render(QPoint(1000,1000),1.0)
#im = np.array(item.image)
#im = cv2.Canny(im.astype("uint8"), 150, 200)
#cv2.imshow("accum", im.astype("uint8"))
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#item.show()
app.exec()