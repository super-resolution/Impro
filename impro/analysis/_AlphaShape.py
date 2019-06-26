"""
====================================================
Experimental class for 3d alpha shapes
====================================================
"""
import numpy as np
from scipy.spatial import Delaunay
from numba import jit
import time


@jit(nopython=True)
def alpha_shape(points, simplices, tri2, lines):
    the_real_triangles = tri2
    for j in range(len(simplices)):
        #simplex = copy.deepcopy(simpl)
        #del simpl
        the_real_triangles[j,0:3] = simplices[j]
        triangle = points[simplices[j]]
        circumcircle = circle_rad(triangle)
        the_real_triangles[j,3] = circumcircle
        for i in range(3):
            p1,p2 = simplices[j,i],simplices[j,(i+1)%3]
            super2 = find_simplices(simplices,p1,p2,j)
            lines[j*3+i] = np.array([simplices[j,i],simplices[j,(i+1)%3],np.float32(j),np.float32(super2),0.0,0.0])
    return lines, the_real_triangles


def alpha_shape3(points, simplices, tet2, tri):
    the_real_triangles = tet2
    for j in range(len(simplices)):
        #simplex = copy.deepcopy(simpl)
        #del simpl
        the_real_triangles[j,0:4] = simplices[j]
        tetrahedron = points[simplices[j]]
        circumcircle = r2_circumsphere_tetrahedron_single(tetrahedron)
        the_real_triangles[j,4] = circumcircle
        for i in range(4):
            p1,p2,p3 = simplices[j,i],simplices[j,(i+1)%4],simplices[j,(i+2)%4]
            super2 = find_tetra(simplices,p1,p2,p3,j)
            tri[j*4+i] = np.array([simplices[j,i],simplices[j,(i+1)%4], simplices[j,(i+2)%4],np.float32(j),np.float32(super2),0.0,0.0])
    return tri, the_real_triangles

def alpha_shape2(points, simplices, tri,):
    for j in range(len(simplices)):
        #the_real_triangles[j,0:3] = simplices[j]
        #tetrahedron = points[simplices[j]]
        #circumcircle = r2_circumsphere_tetrahedron_single(tetrahedron)
        #the_real_triangles[j,4] = circumcircle
        for i in range(3):
            p1,p2 = simplices[j,i],simplices[j,(i+1)%3]
            super2 = find_tri(simplices,p1,p2,j)
            tri[j*3+i] = np.array([simplices[j,i],simplices[j,(i+1)%3], np.float32(j),np.float32(super2[0]), np.float32(super2[1]), np.float32(super2[2]),np.float32(super2[2]) ,0.0,0.0])
    return tri

def merge(simplices, d):
    indices_sorted = np.sort(simplices[:,0:d], axis=1)
    a, index = np.unique(indices_sorted, return_index=True, axis=0)
    simplices = simplices[index]
    return simplices

def finish(lines, the_real_triangles):
    for line in lines:
        if line[-3]==-1:
            line[-1] = 1000000
            line[-2] = the_real_triangles[int(line[-4]),-1]
        else:
            line[-1] = max(the_real_triangles[int(line[-4]),-1], the_real_triangles[int(line[-3]),-1])
            line[-2] = min(the_real_triangles[int(line[-4]),-1], the_real_triangles[int(line[-3]),-1])
    return lines

def finish_lines(lines, the_real_triangles):
    for line in lines:
        for i in range(5):
            if line[2+i] != -1:
                tri = the_real_triangles[line[2+i].astype(np.int32)]
                ma = np.max(tri[5:])
                mi = np.min(tri[5:])
                if ma >line[-1]:
                    line[-1] = ma
                if mi <line[-2]:
                    line[-2] = mi
    return lines

@jit(nopython = True)
def find_tri(simplices,p1,p2,z):
    tri = np.array([-1,-1,-1,-1])
    running = 0
    for i in range(len(simplices)):
        v1 = False
        v2 = False
        for j in range(3):
            x= simplices[i,j] - p1
            if np.abs(simplices[i,j] - p1) < 0.5:
                v1 = True
            if np.abs(simplices[i,j] - p2) < 0.5:
                v2 = True
        if v1 and v2 and i!=z:
            if running >2:
                print("To much triangles")
                break
            tri[running] = i
            running += 1
    return tri

@jit(nopython = True)
def find_simplices(simplices,p1,p2,z):
    for i in range(len(simplices)):
        v1 = False
        v2 = False
        for j in range(3):
            if simplices[i,j] == p1:
                v1 = True
            if simplices[i,j] == p2:
                v2 = True
            if v1 and v2 and i!=z:
                return i
    return -1

@jit(nopython = True)
def find_tetra(simplices,p1,p2,p3,z):
    for i in range(len(simplices)):
        v1 = False
        v2 = False
        v3 = False
        for j in range(4):
            if simplices[i,j] == p1:
                v1 = True
            if simplices[i,j] == p2:
                v2 = True
            if simplices[i,j] == p3:
                v3 = True
            if v1 and v2 and v3 and i!=z:
                return i
    return -1

@jit
def nb_dot(x, y):
    val = 0
    for x_i, y_i in zip(x, y):
        val += x_i * y_i
    return val

@jit
def nb_cross(x, y):
    val = np.array([  x[1]*y[2] - x[2]*y[1],
             x[2]*y[0] - x[0]*y[2],
             x[0]*y[1] - x[1]*y[0] ])
    return val

@jit
def r2_circumsphere_tetrahedron_single(points):
    a, b, c, d = points
    ad = a - d
    bd = b - d
    cd = c - d

    ad2 = nb_dot(ad, ad)
    bd2 = nb_dot(bd, bd)
    cd2 = nb_dot(cd, cd)

    cross_1 = nb_cross(bd, cd)
    cross_2 = nb_cross(cd, ad)
    cross_3 = nb_cross(ad, bd)

    q = ad2 * cross_1 + bd2 * cross_2 + cd2 * cross_3
    p = 2 * np.abs(nb_dot(ad, cross_1))
    if p < 1e-10:
        return np.infty

    r2 = nb_dot(q, q) / p ** 2
    return r2

@jit(nopython = True)
def circle_rad(points):
    d = np.zeros(3)
    s=0
    for i in range(3):
        vec1 = points[i,0] - points[(i+1)%3,0];
        vec2 = points[i,1]-points[(i+1)%3,1];
        d[i] = np.sqrt(vec1*vec1+vec2*vec2);
        s += d[i]
    s = s/2
    area = np.sqrt(s*(s-d[0])*(s-d[1])*(s-d[2]))
    circle_r = d[0]*d[1]*d[2]/(4.0*area)
    return circle_r

def get_simplices(points):
    tri = Delaunay(points)
    simplices = tri.simplices.copy()
    lines = np.zeros([simplices.shape[0]*3,6]).astype(np.float32)
    tri2 = np.zeros([simplices.shape[0], 4]).astype(np.float32)
    lines, the_real_triangles = alpha_shape(points, simplices, tri2, lines)
    lines = finish(lines, the_real_triangles)
    return lines

def get_simplices3(points):
    tri = Delaunay(points)
    simplices = tri.simplices.copy()
    tri = np.zeros([simplices.shape[0]*4,7]).astype(np.float32)
    tet2 = np.zeros([simplices.shape[0], 5]).astype(np.float32)
    triangles, the_real_tetra = alpha_shape3(points, simplices, tet2, tri)
    triangles = merge(triangles,3)
    triangles = finish(triangles, the_real_tetra)
    lin2 = np.zeros([triangles.shape[0] * 3, 9]).astype(np.float32)
    lines = alpha_shape2(points, triangles, lin2)
    lines = merge(lines,2)
    lines = finish_lines(lines, triangles)
    return lines



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    points = np.random.random((8,3))*10
    t1 = time.time()
    lines = get_simplices3(points)

    print(t1-time.time())
    print(lines)
    #for line in lines:
    alpha = 1000
    for line in lines:
        p = points[line[0:2].astype(np.int32)]
        if line[-2]>alpha:
            continue
        if line[-2]<alpha and line[-1]>alpha:
            ax.plot(p[:,0],p[:,1],p[:,2],  color="r", antialiased=True)
        else:
            ax.plot(p[:,0],p[:,1],p[:,2], color="g",
                            antialiased=True)

    # for line in lines:
    #     p = points[line[0:3].astype(np.int32)]
    #     if line[-2]>alpha:
    #         continue
    #     if line[-2]<alpha and line[-1]>alpha:
    #         ax.plot_trisurf(points[line[0:3].astype(np.int32),0],points[line[0:3].astype(np.int32),1],points[line[0:3].astype(np.int32),2],  color=(0,0,0,0),edgecolor='Gray', antialiased=True)
    #     else:
    #         ax.plot_trisurf(points[line[0:3].astype(np.int32), 0], points[line[0:3].astype(np.int32), 1],
    #                         points[line[0:3].astype(np.int32), 2], edgecolor='b',
    #                         antialiased=True)

    #ax.scatter3D(points[:,0],points[:,1],points[:,3])
    plt.show()