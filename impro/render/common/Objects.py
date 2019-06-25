from OpenGL.GL import *
from OpenGL.arrays import vbo
import numpy as np


class Cube:
    """
    ====================================================
    Create a Cube mesh object for OpenGL
    ====================================================
    """
    def __init__(self):
        O = -1.0
        X = 1.0
        positions = np.array([O, O, O, O, O, X, O, X, O, O, X, X, X, O, O, X, O, X, X, X, O, X, X, X,],dtype="f")
        indices = np.array([
        7, 3, 1, 1, 5, 7,
        0, 2, 6, 6, 4, 0,
        6, 2, 3, 3, 7, 6,
        1, 0, 4, 4, 5, 1,
        3, 2, 0, 0, 1, 3,
        4, 6, 7, 7, 5, 4,
        ], dtype=np.int32)
        #Create the VBO for positions:
        self.vertex_vbo = vbo.VBO(data=positions, usage=GL_STATIC_DRAW, target=GL_ARRAY_BUFFER)
        #Create the VBO for indices:
        self.index_vbo = vbo.VBO(data=indices ,usage=GL_STATIC_DRAW, target=GL_ELEMENT_ARRAY_BUFFER)


class UnitCube:
    """
    ====================================================
    Create a Cube mesh object for OpenGL
    ====================================================
    """
    def __init__(self):
        self.vertices = np.array([[-0.5,-0.5,-0.5],
                             [0.5,-0.5,-0.5],
                             [0.5, 0.5,-0.5],
                             [-0.5, 0.5,-0.5],
                             [-0.5,-0.5, 0.5],
                             [0.5,-0.5, 0.5],
                             [0.5, 0.5, 0.5],
                             [-0.5, 0.5, 0.5]],dtype="f")
        self.edge_list = np.array([[ 0,1,5,6, 4,8,11,9, 3,7,2,10 ],
                         [0,4,3,11, 1,2,6,7, 5,9,8,10 ],
                         [1,5,0,8, 2,3,7,4, 6,10,9,11],
                         [7,11,10,8, 2,6,1,9, 3,0,4,5],
                         [8,5,9,1, 11,10,7,6, 4,3,0,2],
                         [9,6,10,2, 8,11,4,7, 5,0,1,3],
                         [9,8,5,4, 6,1,2,0, 10,7,11,3],
                         [10,9,6,5, 7,2,3,1, 11,4,8,0]
                         ], dtype=np.int32)
        self.edges = np.asarray([[0,1],[1,2],[2,3],[3,0],[0,4],[1,5],[2,6],[3,7],[4,5],[5,6],[6,7],[7,4]], dtype=np.int32)


class Quad:
    """
    ====================================================
    Create a Quad mesh object for OpenGL
    ====================================================
    """
    def __init__(self):
        positions = np.array([
        [-1, -1],
         [-1, 1],
        [1,  -1],
         [1,  1],
        ], dtype="f")
        self.vertex_vbo = vbo.VBO(data=positions, usage=GL_STATIC_DRAW, target=GL_ARRAY_BUFFER)


class Texture:
    """
    ====================================================
    Create a Texture mapping for OpenGL
    ====================================================
    """
    def __init__(self):
        positions = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            ], dtype="f")
        self.vertex_vbo = vbo.VBO(data=positions, usage=GL_STATIC_DRAW, target=GL_ARRAY_BUFFER)


class Surface:
    """
    ====================================================
    Create a Surface object for OpenGL
    ====================================================
    """
    def __init__(self):
        positions = np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            ],dtype="f")
        #indices = np.array([
        #    0, 1, 2,
        #    2, 1, 3
        #], dtype=np.int32)
        #Create the VBO for positions:
        self.vertex_vbo = vbo.VBO(data=positions, usage=GL_STATIC_DRAW, target=GL_ARRAY_BUFFER)
        #Create the VBO for indices:
        #self.index_vbo = vbo.VBO(data=indices, usage=GL_STATIC_DRAW, target=GL_ELEMENT_ARRAY_BUFFER)