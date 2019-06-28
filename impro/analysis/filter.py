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
import scipy


class Filter:

    @staticmethod
    def values_filter(value, minimum, maximum):
        """
        Get the indices of an array values with value > minimum and value < maximum

        Parameters
        ----------
        value: np.array
            Array with values to filter
        minimum: float
            minimum value to pass the filter
        maximum: float
            maximum value to pass the filter

        Returns
        -------
        filt: np.array
            indices of entries passing the filter
        """
        print('ValueFilter')
        filt = np.where(np.logical_and(value > minimum, value>maximum))
        return filt.astype(np.int64)

    @staticmethod
    def local_density_filter(points, r, n):
        """
        Get the indices of all points having a minimum of n neighbors in a radius of r

        Parameters
        ----------
        points: np.array
            Points to filter (X,Y,Z)
        r: float
            Search for neighbors in a radius of r
        n: int
            Required neighbors in radius r to pass the filter

        Returns
        -------
        filt: np.array
            indices of entries passing the filter
        """
        print("LdFilter")
        rsd=np.empty((len(points), 3), dtype=np.int)
        rsd[:,0]=np.asarray(points)[:,0]
        rsd[:,1]=np.asarray(points)[:,1]
        rsd[:,2]=np.asarray(points)[:,3]
        filt=[]
        tree=scipy.spatial.cKDTree(rsd)
        Neighbors = tree.query_ball_point(rsd, r, n_jobs=-1)
        for i,j in enumerate(Neighbors):
            if len(j)>n:
                filt.append(i)
        return np.array(filt).astype(np.int64)

