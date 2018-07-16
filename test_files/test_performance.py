import time
import cgalpy
import numpy as np
import processing.CudaAlphaShape as alpha_cuda


def time_CGAL(points, iterations):
    tim = []
    for i in range(iterations):
        z=[]
        t1 = time.time()
        for i in points:
            z.append(cgalpy.Point_2(i[0].astype(np.float64),i[1].astype(np.float64)))
        calpha = cgalpy.Alpha_shape_2(np.int32(16900),z)
        edges = calpha.get_alpha_segments(calpha.Classification_type.REGULAR)
        tim.append(time.time()-t1)
    tim = np.asarray(tim)
    print("Mean: ",np.mean(tim),"Deviation: ", np.std(tim))

def time_cuda_alpha(points, iterations):
    tim = []
    for i in range(iterations):
        simp = alpha_cuda.get_k_simplices(points[...,0:2])
        tim.append(np.array([simp[1], simp[2], simp[3]]))
    tim = np.asarray(tim)
    print("Delaunay, alpha, merge, Mean: ",np.mean(tim, 0),"Deviation: ", np.std(tim, 0))