import numpy as np
import scipy



class filter():
    @staticmethod
    def z_filter(points, min, max):
            print ("ZFilter")
            pointsf1 = points[points[...,3]<=min]
            pointsf2 = pointsf1[pointsf1[...,3]>=max]
            out = np.array([pointsf2])
            return out

    @staticmethod
    def photon_filter(points, min, max):
            print('PhotonFilter')
            pointsf1 = points[points[...,4]>=min]
            pointsf2 = pointsf1[pointsf1[...,4]<=max]
            #out = np.array([pointsf2])
            return pointsf2

    @staticmethod
    def frame_filter(points, min, max):
            print('FrameFilter')
            pointsf1 = points[points[...,6]>min]
            pointsf2 = pointsf1[pointsf1[...,4]<=max]
            #out= np.array(pointsf2)
            return pointsf2

    @staticmethod
    def local_density_filter(points, r, n):
                """filter data for localizations with n neighbors within eps.
                Returns list of indices of included points."""
                print("LdFilter")
            #for k in range(len(points)):
                rsd=np.empty((len(points), 3), dtype=np.int)
                rsd[:,0]=np.asarray(points)[:,0]
                rsd[:,1]=np.asarray(points)[:,1]
                rsd[:,2]=np.asarray(points)[:,3]
                filt=[]
                tree=scipy.spatial.cKDTree(rsd)
                x= tree.query_ball_point(rsd, r, n_jobs=-1)
                for i,j in enumerate(x):
                    if len(j)<=n:
                        filt.append(i)
                #if list(np.delete(np.asarray(points),filt,0))==[]:
                    #outpoints.append(np.asarray([]))
                #else:
                outpoints = np.delete(np.asarray(points),filt,0)
                return outpoints