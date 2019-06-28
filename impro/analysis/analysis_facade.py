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
"""
:Author:
  `Sebastian Reinhard`

:Organization:
  Biophysics and Biotechnology, Julius-Maximillians-University of Würzburg

:Version: 2019.06.26
"""


import numpy as np
import pyqtgraph.opengl as gl
import pyqtgraph as pg
from ..render import render as rend
from ..analysis import alpha_shape_gpu as alpha
from ..analysis.hough_transform_gpu import *
from skimage import transform
from scipy.stats.mstats import pearsonr
from PyQt5.QtCore import QPoint


app = pg.mkQApp()


def create_alpha_shape(storm_data: np.ndarray, alpha_value: float, px_size=32.2, line_width=1.0)-> np.ndarray:
    """
    Facade function to render alpha shape of dSTORM data

    Image is rendered from (0,0) to the maximal dimension of the dSTORM data.

    Parameters
    ----------
    storm_data: np.ndarray(nx5)
        Point Cloud data to be rendered
    px_size: float
        Pixel size for the rendered image in nanometer per pixel
    alpha_value: float
        Core value to compute the alpha shape in nanometer
    line_width: float
        Line width of the alpha shape lines in a.u.


    Returns
    -------
    image: np.array
        Rendered image

    Example
    -------
    >>> points = np.random.randint(0,32000,(1000000, 2))
    >>> image = create_alpha_shape(points, 130)
    >>> image.shape
    (994,994)
    """
    k_simplices = alpha.get_k_simplices(storm_data[...,0:2])[0]

    points = np.empty((2*k_simplices.shape[0],5))
    points[::2,0] = storm_data[(k_simplices[...,0]).astype(np.int32),0]
    points[::2,1] = storm_data[(k_simplices[...,0]).astype(np.int32),1]
    points[1::2,0] = storm_data[(k_simplices[...,1]).astype(np.int32),0]
    points[1::2,1] = storm_data[(k_simplices[...,1]).astype(np.int32),1]
    points[...,2] = np.repeat(k_simplices[...,2],2,)
    points[...,3] = np.repeat(k_simplices[...,3],2,)
    points[...,4] = np.repeat(k_simplices[...,4],2,)

    widget = gl.GLViewWidget()
    widget.show()
    alpha_shape_render = rend.alpha_complex(filename = r"\Alpha")
    sizex = points[:,0].max()/px_size
    sizey = points[:,1].max()/px_size

    alpha_shape_render.set_data(position=points[...,0:2], simplices=points[...,2:5], alpha=np.float64(alpha_value), size=float(line_width))
    alpha_shape_render.background_render(QPoint(sizex, sizey), 1.0)
    image = np.array(alpha_shape_render.image)
    return image


def create_storm(storm_data: np.ndarray, px_size=32.2, size=20, cluster=np.ndarray([])):
    """
    Facade function to render an image of dSTORM data.

    Image is rendered from (0,0) to the maximal dimension of the dSTORM data.

    Parameters
    ----------
    storm_data: np.array(nx5)
        Point Cloud data to be rendered
    px_size: float
        Pixel size for the rendered image in nanometer per pixel
    size: float
        Size of the rendered points in nanometer
    cluster: np.array(n)
        Affiliation of the i th point to cluster[i] cluster


    Returns
    -------
    image: np.array
        Rendered image

    Example
    -------
    >>> points = np.random.randint(0,32000,(1000000, 5))
    >>> image = create_alpha_shape(points, 130)
    >>> image.shape
    (994,994)
    """
    widget = gl.GLViewWidget()
    widget.show()
    sizex = storm_data[...,0].max()/px_size
    sizey = storm_data[...,1].max()/px_size
    point_cloud_render = rend.points(filename=r"\STORM2")
    if cluster.size != 0:
        point_cloud_render.set_data(position=storm_data, size=float(size), maxEmission=256.0, color=np.array([1.0, 0.0, 0.0, 1.0]),
                       cluster=cluster)
    else:
        point_cloud_render.set_data(position=storm_data, size=float(size), maxEmission=256.0, color=np.array([1.0, 0.0, 0.0, 1.0]))
    point_cloud_render.background_render(QPoint(sizex, sizey), 1.0)
    image = np.array(point_cloud_render.image)
    return image


def find_mapping(target: np.ndarray, source_color: np.ndarray, n_col=5, n_row=5, offset=0)-> np.ndarray:
    """
    Facade function to find a mapping from source_color image to gray scale target image.


    Parameters
    ----------
    target: np.array
        Target image of affine transformation
    source_color: np.array
        Source image of affine transformation
    n_col: int
        number of collums to seperate the source image into
    n_row: int
        number of rows to seperate the source image into
    offset: int
        Start seperating the source image at (0+offset,0+offset). Offset is given in pixel


    Returns
    -------
    source_points, target_points: np.array
        Source and target points for affine transformation
    overlay: np.array
        Target image overlayed with source image segments at the matching position
    results: list
        Results of the Weighted General Hough Transform

    Example
    -------
    >>> import cv2
    >>> points = np.random.randint(0,32000,(1000000, 5))
    >>> source = create_alpha_shape(points, 130)
    >>> points = np.random.randint(0,120000,(1000000, 5))
    >>> target = create_alpha_shape(points, 130)
    >>> find_mapping(cv2.cvtColor(target, cv2.COLOR_RGBA2GRAY), source)
    """
    results = []
    target = target.astype(np.uint8)
    #create RGBA image to overlay target and source segments
    overlay = cv2.cvtColor(target, cv2.COLOR_GRAY2RGBA).astype(np.uint16)
    source_gray = cv2.cvtColor(source_color, cv2.COLOR_RGBA2GRAY)

    norm = np.linalg.norm(source_gray)

    ght_target = GHTImage(target, blur=5, canny_lim=(130,180))
    H = HoughTransform()
    H.target = ght_target

    source_points = []
    target_points = []
    for i in range(n_col):
        for j in range(n_row):
            k,l = j*200+offset,200+j*200+offset
            m,n = i*200+offset, 200+i*200+offset
            template = source_gray[k:l,m:n]
            source_points.append(np.array((k+template.shape[0]/2,m+template.shape[1]/2)))

            norm_template = np.linalg.norm(template)
            if norm_template< 0.15*norm:
                results.append([-1,np.array([-1,-1,-1,-1])])
                target_points.append(np.array([0,0]))
                print(f"not enough data for template {i}, {j}")
                continue
            template_color = source_color[k:l,m:n]

            ght_template = TemplateImage(template)
            H.template = ght_template
            res = H.transform()
            results.append(res[0:2])
            target_points.append(res[1][0:2])

            M = cv2.getRotationMatrix2D((template_color.shape[0] / 2, template_color.shape[1] / 2), res[0], 1)
            template_color = cv2.warpAffine(template_color, M, (template_color.shape[0], template_color.shape[1]))
            for h in range(template.shape[0]):
                    for e in range(template.shape[1]):
                        try:
                            overlay[int(2*res[1][0]-template.shape[0]/2+h),int(2*res[1][1]-template.shape[1]/2+e)] += template_color[h,e,0:4]
                        except Exception as error:
                            print('Caught this error: ' + repr(error))
    return source_points,target_points,overlay,results


def error_management(result_list: list, source_points, target_points, n_row = 5):
    """
    Test the results of th weighted GHT for missmatches

    Parameters
    ----------
    result_list: list
        results of the Weighted General Hough Transform
    source_points: np.array
        Source points of affine transformation
    target_points: np.array
        Target points of affine transformation
    n_row: int
        number of rows used for the weighted GHT


    Returns
    -------
    source_points, target_points: np.array
        Filtered source and target points for affine transformation
    """
    num = []
    for i,ent1 in enumerate(result_list):
        if ent1[1][-1] < 0:
            continue
        row1 = int(i/n_row)
        col1 = i % n_row
        max=0
        for j,ent2 in enumerate(result_list):
            if ent2[1][-1]<0:
                continue
            row2 = int(j/n_row)
            col2 = j%n_row
            val1 = ent1[1][0]-(col1-col2)*100*np.cos(np.deg2rad(ent1[0]))+(row1-row2)*100*np.sin(np.deg2rad(ent1[0]))
            val2 = ent1[1][1]-(row1-row2)*100*np.cos(np.deg2rad(ent1[0]))-(col1-col2)*100*np.sin(np.deg2rad(ent1[0]))
            if np.absolute(val1-ent2[1][0])<25 and np.absolute(val2-ent2[1][1])<25:
                max += 1
        if max > 7:
            num.append(i)
    num = np.array(num).astype(np.int32)
    source_points = np.asarray(source_points)
    target_points = np.asarray(target_points)
    source_points = source_points[num]
    target_points = target_points[num]
    source_points = np.fliplr(source_points)
    target_points = np.float32(target_points)*2
    target_points = np.fliplr(target_points)
    return source_points,target_points


def pearson_correlation(target: np.ndarray, source: np.ndarray, map: np.ndarray)-> float:
    """
    Compute pearson correlation index after alignment

    Parameters
    ----------
    target: np.array
        target image in gray scale
    source: np.array
        source image in gray scale
    map: sklearn.transformation
        computed transformation


    Returns
    -------
    source_points, target_points: np.array
        Filtered source and target points for affine transformation

    Example
    -------
    >>> import skimage.transform
    >>> source = np.ones((1000,1000))
    >>> target = np.ones((1000,1000))
    >>> source_points = np.array([[1.0,1.0],[500,500],[700,500])
    >>> target_points = source_points
    >>> M = transform.estimate_transform("affine",source_points,target_points)
    >>> pearson_correlation(target, source, M)
    (1.0)
    """
    mask = np.zeros_like(target)
    mask[0:source.shape[0], 0:source.shape[1]] = 1
    source_extended = np.zeros_like(target)
    source_extended[0:source.shape[0], 0:source.shape[1]] = source

    binary_mask = transform.warp(mask, inverse_map=map.inverse).astype(np.bool)

    source_warped = transform.warp(source_extended, inverse_map=map.inverse)*255
    masked_source = np.ma.array(data=source_warped, mask=np.logical_not(binary_mask))

    masked_target = np.ma.array(data=target, mask=np.logical_not(binary_mask))

    corr_coef = pearsonr(masked_source.flatten(),masked_target.flatten())


    print("image:",corr_coef)#masked correlation only compare visible parts

    return corr_coef