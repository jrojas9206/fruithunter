import os 
import sys 
import glob 
import math 
import argparse 
import numpy as np 
import open3d as o3d
from sklearn.neighbors import KDTree

class PointCloudFEvaluator(object):
    
    """
    Class to evluate and compare the point clouds 
    """

    def pointCloud_generalDescription(self, fileLst, pcformat="txt", geometry="cylinder", verbose=False):
        """
        Load the point clouds of the given folder and extract the few descriptors of the geometry   
        :INPUT:
            fileLst: str, List of the files to verify 
            pcformat: Format of the point cloud to load, default 'txt'
            geometry: Figure from where must verified the area, cylinder or cube
            verbose: show few messages of the followed steps by the function, default 'False'
        :OUTPUT:
            dict, {fileName: {base_geometry:, volume:, height:, x: [min, max], y: [min, max], z: [min, max], fileSz: }} 
        NOTE: The units of the returned values depend on the units of the point cloud
        """
        baseGeom = {}
        areaBck = {}
        g2ret = {}
        l2ret = []
        for idx, fname in enumerate(fileLst, start=1): 
            # Get the name of the file from the given path 
            lst_path = os.path.split(fname)
            a_file_name = lst_path[-1].split('.')
            if(verbose):
                print("-> Loading[%s/%s]: %s"%(idx,len(fileLst), a_file_name[0]), end="\r" if idx<len(fileLst) else "\n")
            a_pc = np.loadtxt(fname)
            pc_shape = a_pc.shape
            a_sz = os.path.getsize(fname)/(1024*1024) # Get the size of the file in MB
            # Main geometry positions
            min_pos_x = np.min(a_pc[:,0])
            max_pos_x = np.max(a_pc[:,0])
            cx = (abs(min_pos_x)+abs(max_pos_x))/2.0 # Center in the X axis 
            min_pos_y = np.min(a_pc[:,1])
            max_pos_y = np.max(a_pc[:,1])
            cy = (abs(min_pos_y)+abs(max_pos_y))/2.0 # Center in the Y axis
            min_pos_z = np.min(a_pc[:,2])
            max_pos_z = np.max(a_pc[:,2])
            cz = (abs(min_pos_z)+abs(max_pos_z))/2.0 # Center in the Z axis
            # Remove the Annotations column and convert to Open3D format
            #pcd = o3d.geometry.PointCloud()
            #pcd.points = o3d.utility.Vector3dVector(a_pc[:,0:3])
            #a_cHull, _ = o3d.open3d.geometry.PointCloud.compute_convex_hull(pcd)
            # Visualice Convex hull
            #a_cHull.paint_uniform_color((1, 0, 0))
            #o3d.visualization.draw_geometries([pcd, a_cHull])
            # Get the min and the max of the X, Y, Z coordinates and with this points estimate the 
            # radius or the square that cover the given geometry 
            if(geometry=="cylinder"):
                # Radius over the axis Y and Z
                radiusOnX = abs( (abs(max_pos_x)-cx) )
                radiusOnY = abs( (abs(max_pos_y)-cy) )
                # Retrive the biggest radius  
                R = max(radiusOnX, radiusOnY)
                # 
                c_volume = abs((math.pi*(R**2)*max_pos_z)) # Cylinder volume
                g2ret = {"geometry": "cylinder", "radius": R, "center": [cx, cy], "height": max_pos_z, "volume": c_volume, "filename": a_file_name[0], "fileSize": a_sz, "pcShape": pc_shape}
            elif(geometry=="cube"):
                bottom_left = [min_pos_x, max_pos_y]
                top_right   = [max_pos_x, min_pos_y]
                s_volume = (abs(min_pos_x)-abs(max_pos_x))*(abs(min_pos_y)-abs(max_pos_y)*max_pos_z) # cube volume 
                g2ret = {"geometry": "square", "bottom_left": bottom_left, "top_right": top_right, "volume": abs(s_volume), "filename": a_file_name[0], "fileSize": a_sz, "pcShape":pc_shape}
            else: 
                return {}
            # Keep the biggest area 
            if(idx==1): # Start the area variable 
                areaBck = g2ret.copy()
            #if(verbose):
            #    print("-------")
            #    print("BCK: %s" %( str(areaBck) ))
            #    print("Actual: %s" %( str(g2ret) ) )
            #    print("--------")
            if(g2ret["volume"] > areaBck["volume"]):
                areaBck = g2ret
            l2ret.append(g2ret)
        return l2ret, areaBck

    def get_volume(self, pc, geometry="cylinder"):
        """
        Estimate the min volume that enclose a point clouds 
        """
        c_volume = 0.0
        min_pos_x = np.min(pc[:,0])
        max_pos_x = np.max(pc[:,0])
        cx = (abs(min_pos_x)+abs(max_pos_x))/2.0 # Center in the X axis 
        min_pos_y = np.min(pc[:,1])
        max_pos_y = np.max(pc[:,1])
        cy = (abs(min_pos_y)+abs(max_pos_y))/2.0 # Center in the Y axis
        min_pos_z = np.min(pc[:,2])
        max_pos_z = np.max(pc[:,2])
        cz = (abs(min_pos_z)+abs(max_pos_z))/2.0 # Center in the Z axis
        if(geometry=="cylinder"):
            # Radius over the axis Y and Z
            radiusOnX = abs( (abs(max_pos_x)-cx) )
            radiusOnY = abs( (abs(max_pos_y)-cy) )
            # Retrive the biggest radius  
            R = max(radiusOnX, radiusOnY)
            # 
            c_volume = abs((math.pi*(R**2)*max_pos_z)) # Cylinder volume
        elif(geometry=="cube"):
            bottom_left = [min_pos_x, max_pos_y]
            top_right   = [max_pos_x, min_pos_y]
            c_volume = (abs(min_pos_x)-abs(max_pos_x))*(abs(min_pos_y)-abs(max_pos_y)*max_pos_z) # cube volume 
        return c_volume

    def get_density_fromBaseGeometry(self, pc, baseVolume):
        """
        Estimate the density of point in given are

        :INPUT:
            pointCloud: numpy array, (N,3)
            baseVolume: float32
        :OUTPUT:
            float32
        """
        return pc.shape[0]/baseVolume

    def get_point_sparcing(self, pc, pointDensity):
        """

        """
        return math.sqrt(1/pointDensity)

    def get_point_cloud_avg_density(self, pc, radius=0.1):
        """
        Get the point density of a point cloud from several 
        density measurements arround the point cloud  
        
        Npoints/R**2? -- must be Npoints/ 4/3*pi*R**3?

        :INPUT:
            pc: numpy array (N,3)
            radius: Distance to evaluate the nearest points, sphere radius 
        :OUTPUT:
            float32
        """
        # Get the KDtree representation of the point cloud 
        # and from it begin to estimate the densities 
        # base on the number of points inside the 
        # evaluate sphere 

        kdt_rep = KDTree(pc, leaf_size=2)
        lstDens = []
        for idx, _ in enumerate( range(pc.shape[0]) ):
            nPts = kdt_rep.query_radius(pc[idx].reshape(1,-1), r=radius, count_only=True)
            lstDens.append( float(nPts/( radius**2 )) )
        return lstDens

    def experiment_mean_density(self, pcStrList, lstPC_gdescriptor):
        """
        Get the avarange density of a set of point clouds 
        :INPUT:
            pcStrList: list with the name of the files to evaluate
            lstPC_gdescriptor: List of dict, it must contain a key called {volume} 
        :OUTPUT:
            float32
        """
        a2ret = 0
        for idx, (fname, adict) in enumerate( zip(pcStrList, lstPC_gdescriptor), start=1):
            a_pc = np.loadtxt(fname)
            a2ret += self.get_density_fromBaseGeometry(a_pc, adict["volume"])
        return float(a2ret)/len(pcStrList)


    def sort_tree_by(self, LstOfDict, sortBy="volume"):
        """
        Sort the trees by their estimated volume 
        
        :Algorithm: 
            Insertion Sort 
        :INPUT:
            LstOfDict: List of dictionaries, The dictionary must have the "sortBy" key inside {sortBy}
            sortBy: str, Key in the dictionary that is going to be use to sort 
        :OUTPUT:
            Sorted List 
        :NOTE:
            Use with lists of less than 200 positions, after this value it could be little bit slow 
        """
        A = LstOfDict
        for j in range(1,len(LstOfDict), 1, 200):
            key = A[j]
            # Insert  into the sorted sequence 
            i = j-1 
            while( i>-1 and A[i][sortBy]>key[sortBy] ):
                A[i+1] = A[i]
                i = i-1
            A[i+1] = key
        return A

def main():
    parser = argparse.ArgumentParser("Example of the use of the class PointCloudFEvaluator")
    parser.add_argument("path2folder", type=str, help="Path to the folder with the point clouds")
    parser.add_argument("--writeRepor", type=bool, help="If true, a csv file with the global report, it will be written", default=False)
    parser.add_argument("--pcFormat", type=str, help="Format of the point clouds", default="txt")
    args = parser.parse_args()
    oEx = PointCloudFEvaluator()
    return 0

if(__name__ == "__main__"):
    sys.exit(main())