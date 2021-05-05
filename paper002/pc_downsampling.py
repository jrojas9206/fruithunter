import numpy as np 
import random
import math 
import scipy 
from sklearn.metrics.pairwise import pairwise_distances

class Downsample(object):
    """
    Class that contain different methods of subsampling 
    for pointclouds, all the implementation are based only 
    on the CPU
    """

    def decimation(self, pointCloud, npoints=1000, verbose=False):
        """
        Subsample the point cloud using a fix interval based on the number of points that 
        want to be keep at the end.

        :INPUT:
            pointCloud: Numpy array (N,M), point cloud 
            npoints: Intenger, number of points to conserve 
            verbose: print some info
        :OUTPUT:
            Numpy array (N', M)
        ref: https://towardsdatascience.com/how-to-automate-lidar-point-cloud-processing-with-python-a027454a536c#:~:text=%20How%20to%20automate%20LiDAR%20point%20cloud%20sub-sampling,We%20will%20focus%20on%20decimation%20and...%20More%20
        """
        factor = int(math.ceil(pointCloud.shape[0]/npoints))
        if(verbose):
            print("  ->> Decimation factor: %i" %factor)
        points = pointCloud[::factor] 
        return points

    def Greedy_downsampling(self, X, Npts=1000):
        """
        A Naive O(N^2) algorithm to do furthest points sampling
        
        :INPUT:
            X : numpy array (N,N)
        :OUTPUT:
            numpy array
        
        Author: Chris Tralie
        Link: https://gist.github.com/ctralie/
        """
        D = pairwise_distances(X, metric='euclidean')

        N = D.shape[0]
        #By default, takes the first point in the list to be the
        #first point in the permutation, but could be random
        perm = np.zeros(N, dtype=np.int64)
        lambdas = np.zeros(N)
        ds = D[0, :]
        for i in range(1, N):
            idx = np.argmax(ds)
            perm[i] = idx
            lambdas[i] = ds[idx]
            ds = np.minimum(ds, D[idx, :])
        # Get the points 
        _x2r =  X[perm[0:Npts], 0]
        _y2r =  X[perm[0:Npts], 1]
        _z2r =  X[perm[0:Npts], 2]
        # merge
        lst2ret = [] 
        for x,y,z in zip( _x2r, _y2r, _z2r ):
            lst2ret.append(  np.array( [ x,y,z ] )  )
        return lst2ret

    def voxel_subsampling(self, points, voxel_size=0.1, base="center"):
        """
        Subsampling base on voxel definition
        :INPUT:
            pointCloud: numpy array(NxM)
            voxel_size: integer, units of the cloud 
            base: what point will be taken to generate the downsampled cloud
        :OUTPUT:
            numpy array(N',M)
        ref: https://towardsdatascience.com/how-to-automate-lidar-point-cloud-processing-with-python-a027454a536c#:~:text=%20How%20to%20automate%20LiDAR%20point%20cloud%20sub-sampling,We%20will%20focus%20on%20decimation%20and...%20More%20
        """
        # 
        voxel_grid={}
        pc2ret=[]
        last_seen=0
        # Create the voxels
        nb_vox=np.ceil((np.max(points, axis=0) - np.min(points, axis=0))/voxel_size)
        #
        non_empty_voxel_keys, inverse, nb_pts_per_voxel = np.unique(((points - np.min(points, axis=0)) // voxel_size).astype(int), axis=0, return_inverse=True, return_counts=True)
        idx_pts_vox_sorted=np.argsort(inverse)
        # 
        for idx,vox in enumerate(non_empty_voxel_keys):
            voxel_grid[tuple(vox)]= points[idx_pts_vox_sorted[last_seen:last_seen+nb_pts_per_voxel[idx]]]
            if(base=="center"):
                pc2ret.append(np.mean(voxel_grid[tuple(vox)],axis=0))
            elif(base=="bcenter"):
                pc2ret.append(voxel_grid[tuple(vox)][np.linalg.norm(voxel_grid[tuple(vox)] - np.mean(voxel_grid[tuple(vox)],axis=0),axis=1).argmin()])
            else:
                raise IOError("Unknown option")
            last_seen+=nb_pts_per_voxel[idx]
        return np.array(pc2ret)

def random_split(npArray, groups=2):
    """
    Split  the array on several batches 
    :INPUT:
        npArray: Numpy array (NxM)
        groups : Integer, batches to create 
    :OUTPUT:
        List of batches 
    """
    factor = int(npArray.shape[1]/groups)


    