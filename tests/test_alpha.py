import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#points = np.random.randn(100, 2)*100




class alpha_complex():
    def __init__(self, points):
        self.points = points
        self.triangulation = Delaunay(points)
        self.d_simplices = []
        for simpl in self.triangulation.simplices.copy():
            self.d_simplices.append(simplex(simpl,2, self.__calc_sigma(2,simpl)))
        self.neighbors = self.triangulation.neighbors.copy()
        self.k_simplices = []
        for i,neighbor in enumerate(self.neighbors):
            for j in range(3):
                if neighbor[j] != -1:
                    self.k_simplices.append(self.merge(self.d_simplices[i], self.d_simplices[neighbor[0]], j))
                    for k in range(3):
                        self.neighbors[neighbor[j]] = -1#add simplex only once

    def __calc_sigma(self, dim, indices):
        if dim ==2:
            dist = []
            s=0
            for i in range(3):
                d = np.linalg.norm(self.points[indices[i]]- self.points[indices[(i+1)%3]])
                s += d
                dist.append(d)
            #heron formel
            s= s/2
            area = np.sqrt(s*(s-dist[0])*(s-dist[1])*(s-dist[2]))
            circ_r = dist[0]*dist[1]*dist[2]/(4.0*area)
            return circ_r
        elif dim ==1:
            return np.linalg.norm(self.points[indices[0]]- self.points[indices[1]])

    def merge(self, d_sim1,d_sim2,index):
        indices = list([d_sim1.indices[index],d_sim1.indices[(index+1)%3]])
        ksim = simplex(indices,1, self.__calc_sigma(1, indices))
        ksim.super = list([d_sim1,d_sim2])
        ksim.a = min(d_sim1.c, d_sim2.c)
        ksim.b = max(d_sim1.c, d_sim2.c)
        return ksim

class simplex(alpha_complex):
    def __init__(self, indices, dim, sigma):
        self.sub = []
        self.super = []
        self.indices = indices
        self.dim = dim
        self.sigma = sigma
        self.a = 0 #single alpha exposed
        self.b = 0 #double alpha exposed
        self.c = self.sigma
        self.exterior = False
        self.interior = False
        self.surface = False
        self.line = False

    def set_alpha(self, alpha):
        if alpha < self.c:
            self.exterior = True
        elif alpha > self.b:
            self.line = True
        elif alpha > self.a:
            self.surface = True
        else:
            self.interior = True
