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
"""The General Hough Transform (GHT) maps the orientation of edge points in a template image to a predefined origin
(typically the middle pixel of the image). Comparing this map with an image gives each point a rating of likelihood for
being a source point of that map.

:Author:
  `Sebastian Reinhard`

:Organization:
  Biophysics and Biotechnology, Julius-Maximillians-University of Würzburg

:Version: 2019.06.26

Example
-------
>>> target = cv2.imread("path_to_file")
>>> template = cv2.imread("path_to_file")
>>> ght_target = GHTImage(target, blur=5, canny_lim=(130,180))
>>> H = HoughTransform()
>>> H.target = ght_target
>>> ght_template = TemplateImage(template)
>>> H.template = ght_template
>>> res = H.transform()


"""

import pycuda
import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
import cv2
from pycuda.compiler import SourceModule
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
with open(dir_path+ r'/cuda_files/hough_transform.cu', 'r') as f:
    cuda_code = f.read()
mod = SourceModule(cuda_code)


class GHTImage:
    """
    GHT specific image preprocessing

    Attributes
    ----------
    o_image: np.array
        Input image
    image: np.array
        Resized and blurred input image
    canny: np.array
        Canny Edge processed image
    gradient: np.array
        Gradient image

    """
    def __init__(self, image, blur=4, canny_lim=(130,200)):
        self._canny_lim = canny_lim
        self._blur = blur
        self._scale = 0.5
        self.o_image = image
        self._create_images()


    def _create_images(self):
        """
        Create different image types by convolution
        """
        self.image = cv2.resize(self.o_image, (0,0), fx=self._scale, fy=self._scale, interpolation=cv2.INTER_CUBIC)
        self.image = cv2.blur(self.image, (self._blur,self._blur))
        self.canny = cv2.Canny(self.image, self._canny_lim[0],self._canny_lim[1])
        self.gradient = self._create_gradient(self.canny)

    @staticmethod
    def _create_gradient(image):
        """
        Convolve an image with Sobel Kernels in X and Y direction to create a gradient image.
        (Gradient orientation of a box size 5x5 in rad)
        """
        X = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5)
        Y = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=5)
        gradient = np.arctan2(X,Y)
        return gradient


class TemplateImage(GHTImage):
    """
    Extend GHTImage by templte properties

    Attributes
    ----------
    o_image: np.array
        Input image
    image: np.array
        Resized and blurred input image
    canny: np.array
        Canny Edge processed image
    gradient: np.array
        Gradient image
    r_matrix_zero: np.array
        Vector mapping of edge points to a predefined origin
    r_matrix: np.array
        Rotated r_matrix_zero

    """

    def __init__(self, image, **kwargs):
        super().__init__(image, **kwargs)
        self.r_matrix_zero = self._create_r_matrix()
        self.r_matrix = np.array([])

    def _create_r_matrix(self):
        """
        Create R-matrix from gradient image

        Returns
        -------
        np.array
            R-Matrix
        """
        origin = np.asarray((self.gradient.shape[0] / 2, self.gradient.shape[1] / 2))
        Rtable = []
        phitable = []
        [phitable.append([]) for i in range(9)]
        [Rtable.append([]) for i in range(9)]
        for i in range(self.gradient.shape[0]):
            for j in range(self.gradient.shape[1]):
                if self.canny[i, j] == 255:
                    phi = self.gradient[i, j]
                    slice = self._is_slice(phi)
                    phitable[slice].append(phi)
                    Rtable[slice].append(np.array((origin[0] - i + 1, origin[1] - j + 1)))

        self.phi_table = phitable
        return self._table_to_matrix(Rtable)

    def rotate_r_matrix(self, angle):
        """
        Rotate R-Matrix by angle rad

        Params
        -------
        angle: float
            Angle to rotate matrix in rad
        """
        s = np.sin(angle)
        c = np.cos(angle)
        new_table = []
        phitable = []
        [phitable.append([]) for i in range(9)]
        [new_table.append([]) for i in range(9)]
        for i, islice in enumerate(self.phi_table):
            for j, phi in enumerate(islice):
                vec = self.r_matrix_zero[i, j]
                phi_new = phi + angle
                slice = self._is_slice(phi_new)
                rotated = (int(round(c*vec[0]-s*vec[1])),int(round(s*vec[0]+c*vec[1])))
                new_table[slice].append(rotated)
                phitable[slice].append(phi_new)
        self.r_matrix = self._table_to_matrix(new_table)

    @staticmethod
    def _table_to_matrix(table):
        """
        Convert table with different column lenghts to matrix. Added entries are filled with zeros

        Params
        -------
        table: list
            table to convert to numpy array
        Returns
        -------
        np.array
            Matrix
        """
        maximum = 0
        for i in table:
            if len(i) > maximum:
                maximum = len(i)
        R_matrix = np.zeros([9,maximum,2])
        for i,j in enumerate(table):
            for h,k in enumerate(j):
                R_matrix[i][h] = k
        return R_matrix

    @staticmethod
    def _is_slice(phi):
        return int(8 * (phi + np.pi) / (2 * np.pi))


class HoughTransform:
    """
    Perform Gradient Weighted General Hough Transform on the Graphics Processing Unit

    Attributes
    ----------
    rotation_min: float
        Start matching template at rotation_min
    rotation_max: float
        End matching template at rotation_max
    template: np.array
        Matching template image on target image
    target: np.array
        Matching template image on target image
    weighted: bool
        Weight the GHT by Gradient density


    Methods
    -------
    transform()
        Perform GHT algorithm with given data
    """
    def __init__(self, rotation_min=-10, rotation_max=10):
        self.rotation = np.array([rotation_min, rotation_max]).astype(np.int16)
        self.weighted = True
        self.r_table = []
        self.gradient_image = []
        self.create_accum = mod.get_function("create_accum")

    @property
    def rotation_min(self):
        return self.rotation[0]

    @rotation_min.setter
    def rotation_min(self, value):
        if value>=self.rotation[1]:
            raise ValueError(f"Minimum rotation {rotation_min} should be smaller than maximum rotation {rotation_max}")
        self.rotation[0] = value

    @property
    def rotation_max(self):
        return self.rotation[1]

    @rotation_max.setter
    def rotation_max(self, value):
        if value<=self.rotation[1]:
            raise ValueError(f"Minimum rotation {rotation_min} should be smaller than maximum rotation {rotation_max}")
        self.rotation[1] = value

    @property
    def template(self):
        return self._templ

    @template.setter
    def template(self, templ):
        if not isinstance(templ, TemplateImage):
            raise ValueError("Template has to be an instance of class TemplateImage")
        self._templ = templ

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, target):
        if not isinstance(target, GHTImage):
            raise ValueError("Template has to be an instance of class GHTImage")
        self._target = target
        self.weight_array = cv2.boxFilter(self._target.canny.astype(np.uint16) / 255, -1, (100, 100), normalize=False)

    def _fast_weighted_maximas(self, accum, ratio=0.8):
        maxindex = np.unravel_index(accum.argmax(), accum.shape)
        candidates = np.argwhere(accum >= (accum[maxindex]*ratio))
        result = []
        accum_max = accum[candidates[...,0],candidates[...,1]]
        weight = self.weight_array[candidates[...,0],candidates[...,1]]
        for i,candidate in enumerate(candidates):
            result.append(np.array((candidate[0], candidate[1], accum_max[i], weight[i], 10000*accum_max[i]/weight[i]+6*accum_max[i])))
        result = np.asarray(result).astype(np.int32)
        return result

    def transform(self):
        accum = np.zeros_like(self._target.gradient)

        #allocate memmory for matrices

        #Pass target gradient image to GPU
        gpu_gradient_image = Matrix(self._target.gradient.astype(np.float32))


        max_threads = pycuda.tools.DeviceData(dev=None).max_threads
        n = max_threads/1024
        if n < 1:
            raise(EnvironmentError("Upgrade GPU"))
        block_size = (32,32,int(n))

        res = 0,np.zeros(5)

        for i in range(self.rotation[1]-self.rotation[0]):
            angle = i+self.rotation[0]
            self._templ.rotate_r_matrix(np.pi*(angle)/180)
            #self.r_table = self._rot_r_table(np.pi*(angle)/180)

            #Pass empty accumulator array to GPU
            gpu_accumulator_array = Matrix(accum.astype(np.int32))
            try:
                # Pass r-table to GPU
                gpu_r_table = Matrix(self._templ.r_matrix.astype(np.int32))
            except:
                print("Probably r-table empty better check")
                break

            #Compute
            grid = int(self._target.gradient.shape[0]/block_size[0])+0,int(self._target.gradient.shape[1]/block_size[1])+0,int(self._templ.r_matrix.shape[1])
            self.create_accum(gpu_accumulator_array.ptr, gpu_r_table.ptr, gpu_gradient_image.ptr,  block=block_size, grid=grid)
            acc = gpu_accumulator_array.get()

            #Weight or not weight accunulator array
            if self.weighted:
                weighted_acc = self._fast_weighted_maximas(acc, ratio=0.8)
            else:
                weighted_acc = acc
            #Find maximum values(result)
            x = np.unravel_index(weighted_acc[...,4].argmax(),weighted_acc.shape[0])
            if weighted_acc[x,4]>res[1][4]:
                res = (angle,weighted_acc[x])

        return res

class Matrix:
    """
    Wrapper class for matrix and matrixf struct on GPU:
    """
    def __init__(self, array):
        mem_size = 16 + np.intp(0).nbytes
        self.ptr = drv.mem_alloc(mem_size)
        self.data = drv.to_device(array)
        self.shape, self.dtype = array.shape, array.dtype
        self.width = array.shape[1]
        self.height = array.shape[0]
        self.stride = np.int32(0).nbytes
        drv.memcpy_htod(int(self.ptr), np.int32(self.width))
        drv.memcpy_htod(int(self.ptr)+4, np.int32(self.height))
        drv.memcpy_htod(int(self.ptr)+8, np.int32(self.stride))
        drv.memcpy_htod(int(self.ptr)+16, np.intp(int(self.data)))
    def get(self):
        #drv.memcpy_dtoh(array, self.data)
        return drv.from_device(self.data, self.shape, self.dtype)


