import numpy as np
import scipy


class Filter:

    @staticmethod
    def values_filter(value, minimum, maximum):
        """
        ================================================================================
        Get the indices of an array values with value > minimum and value < maximum
        ================================================================================

        Parameters
        ----------
        value: np.array
            Array with values to filter
        minimum: float
            minimum value to pass
        maximum: float
            maximum value to pass

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
        ================================================================================
        Get the indices of all points having a minimum of n neighbors in a radius of r
        ================================================================================

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

