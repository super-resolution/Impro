# Copyright (c) 2018-2022, Sebastian Reinhard
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holders nor the names of any
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

r"""The objective of the alpha shape algorithm is to deliver a formal meaning for the geometric notation of 'shape',
in the context of finite point sets.
The straciatella ice cream example gives a visual explanation of the concept:
Consider a ball of ice cream with chocolate pieces.
The chocolate pieces represent the distribution of points in space (ice cream).
The alpha shape algorithm now tries to eat as much ice cream as possible without touching any chocolate pieces,
using an arbitrary predefined spoon size, the :math:`\alpha`-value. Choosing a very small spoon size results in all ice cream
being eaten, a very big spoon size in no ice cream being eaten at all (Convex Hull). But a size in between creates a
concave hull representing the shape of the distributed chocolate pieces, the desired alpha shape.

.. figure:: fig/alpha.png

   Considering the red dotted line *l*, the limit *a* is defined as the minimum of the diameter of the circumcircles around
   :math:`\triangle_U` and :math:`\triangle_V`. The limit *b* is the maximum of circumcircles around :math:`\triangle_U` and :math:`\triangle_V`.
   Since the :math:`\alpha`-ball is smaller than *b*, but bigger than *a*, *l* is classified as boundary.

References
----------
(1) Edelsbrunner, Herbert ; Mücke, Ernst P.: Three-dimensional Alpha
    Shapes. In: ACM Trans. Graph. 13 (1994), Januar, Nr. 1, 43–72. http:
    //dx.doi.org/10.1145/174462.156635. – DOI 10.1145/174462.156635.
    – ISSN 0730–0301
(2) Fischer, Kaspar: Introduction to Alpha Shapes. https:
    //graphics.stanford.edu/courses/cs268-11-spring/handouts/
    AlphaShapes/as_fisher.pdf.

:Author:
  `Sebastian Reinhard`

:Organization:
  Biophysics and Biotechnology, Julius-Maximillians-University of Würzburg

:Version: 2018.03.09



Example
-------

>>> data = np.random.randint(0,32000,(1000000, 2))
>>> k_simplices = get_k_simplices(data[...,0:2])[0]

"""

import time
import pycuda
import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule
from scipy.spatial import Delaunay
import os
#GPU code to construct d = 2 simplices from delaunay triangulation
dir_path = os.path.dirname(os.path.realpath(__file__))
with open(dir_path+ r'/cuda_files/alpha_shape.cu', 'r') as f:
    cuda_code = f.read()
mod = SourceModule(cuda_code)

class AlphaComplex():
    """
    Derive 2D alpha complex from scipy.spatial Delaunay

    Class to create a alpha complex structure on GPU.


    Parameters
    ----------
    struct_ptr: int
        pointer to allocated memmory for structure
    indices: np.array
        nx3 array containing point indices of the delaunay triangulation simplices
    neighbors: np.array
        nx3 array containing the indices of neighboring simplices
    points: np.array
        nx2 array of the points used for delaunay triangulation
    """
    # size of pointers in struct = memory size needed
    memsize = 4* np.intp(0).nbytes

    def __init__(self, struct_ptr: int, indices: np.ndarray, points: np.ndarray, neighbors: np.ndarray):
        # pointer to allocated memmory
        self.struct_ptr = int(struct_ptr)
        # indices per simplex
        self.indices = indices.astype(np.int32)
        # empty array for k_simplices
        self.k_simplices = np.zeros((neighbors.shape[0]*3,5)).astype(np.float32)
        # neighboring simplices
        self.neighbors = neighbors.astype(np.int32)
        # list of triangulation points
        self.points = points.astype(np.float32)
        # copy arrays to device get pointers for struct
        self.indices_ptr = drv.to_device(self.indices)
        self.points_ptr = drv.to_device(self.points)
        self.neighbor_ptr = drv.to_device(self.neighbors)
        self.k_simplices_ptr = drv.to_device(self.k_simplices)

        # create struct from pointers
        drv.memcpy_htod(self.struct_ptr, np.intp(int(self.indices_ptr)))
        # sizeof(pointer) offset per element
        drv.memcpy_htod(self.struct_ptr+np.intp(0).nbytes, np.intp(int(self.points_ptr)))
        drv.memcpy_htod(self.struct_ptr+np.intp(0).nbytes*2, np.intp(int(self.neighbor_ptr)))
        drv.memcpy_htod(self.struct_ptr+np.intp(0).nbytes*3, np.intp(int(self.k_simplices_ptr)))




    def get(self):
        """
        Returns
        -------
        numpy.array
            nx5 of d=1 simplices
            containing: [index1, index2, dist, sigma1, sigma2] with sigma 1 < sigma 2
        """
        self.result = drv.from_device(self.k_simplices_ptr, self.k_simplices.shape, np.float32)
        return self.result

    def merge(self):
        indices = self.result[...,0:2].astype(np.int32)
        indices_sorted = np.sort(indices,axis=1)
        a,index = np.unique(indices_sorted,return_index=True, axis=0)
        merged = self.result[index]
        return merged

def get_k_simplices(points: np.ndarray):
    """
    Parameters
    ----------
    points: np.array
        nx2 array of points to use for alpha complex

    Returns
    -------
    alpha complex: mx5 array of d=1 simplices containing the upper and lower limits for a simplice to be interior/
    on boundary of the alpha shape.
    """
    t1 = time.time()
    tri = Delaunay(points)
    _tdel = time.time()-t1
    print("Delaunay " + str(points.shape[0]) + " points in " + str(_tdel) + " seconds")
    t1 = time.time()
    simplices = tri.simplices.copy()
    neighbors = tri.neighbors.copy()

    alpha_complex_ptr = drv.mem_alloc(AlphaComplex.memsize)

    alpha_comp = AlphaComplex(alpha_complex_ptr,simplices,points, neighbors)

    func = mod.get_function("create_simplices")

    func(alpha_complex_ptr, block=(500,1,1), grid=(int(simplices.shape[0]/500),1,1))

    alpha_comp.get()
    _talph = time.time()-t1
    print("created alpha complex of " + str(points.shape[0]) + " points in " + str(_talph) + " seconds")
    res = alpha_comp.merge()
    _tmerg = time.time()-t1
    print("merging needs: " + str(_tmerg) + " seconds")
    return res ,_talph, _tdel, _tmerg

if __name__ == "__main__":
    points = (np.random.randn(1000000, 2)*100).astype(np.float32)
    tri = Delaunay(points)
    simplices = tri.simplices.copy()
    neighbors = tri.neighbors.copy()
    alpha_complex_ptr = drv.mem_alloc(AlphaComplex.memsize)

    alpha_comp = AlphaComplex(alpha_complex_ptr, simplices, points, neighbors)

    func = mod.get_function("create_simplices")
    func(alpha_complex_ptr, block=(500,1,1), grid=(int(simplices.shape[0]/500),1,1))
    a = alpha_comp.get()
    drv.stop_profiler()

