import os 
import sys 
import glob 
import sklearn
import argparse 
import numpy as np 
import pandas as pd
import sklearn
import sklearn.cluster
from functools import partial
from multiprocessing import Pool
from config import DBSCAN_lowres
from config import DBSCAN_highres

#EXAMPLE_PATH = "/media/juan/Maxtor/PhD/230818_predictions_randlanet_realdata_2018_2019/merged_230823_prediction_sm_real_scan_only_xyz_2018/"

def load_pointcloud(a_file=list, verbose=False)->(str, np.array):
    """
    Load pointcloud 

    :param a_file: str, String with the file to load
    :return:
        str, numpy_array 
    """
    try:
        pointcloud = np.loadtxt(a_file)
    except ValueError:
        pointcloud = np.loadtxt(a_file, delimiter=",")
    if verbose:
        print("    -> Pointcloud shape: %s" %(str(pointcloud.shape)))
    return os.path.split(a_file)[-1], pointcloud

def filtering(point_cloud, cluster_eps, cluster_size):

    db = sklearn.cluster.DBSCAN(min_samples=1,
                                eps=cluster_eps,
                                metric="euclidean",
                                algorithm="ball_tree").fit(point_cloud)

    nb_cluster = max(db.labels_)

    for i in range(nb_cluster + 1):
        cond = db.labels_ == i
        if np.count_nonzero(cond) < cluster_size:
            db.labels_[cond] = -1

    # return point_cloud[db.labels_ >= 0]
    return db.labels_ >= 0


def clustering(pc, min_samples, eps, filter_cluster_size=None, filter_cluster_eps=None, leafSize=18):

    labels = -1 * np.ones((pc.shape[0], 1), dtype=np.int8)
    cond = np.full((pc.shape[0], ), True)

    if pc.shape[0] > 0:

        if filter_cluster_size is not None and filter_cluster_eps is not None:
            cond = filtering(pc, 
                             filter_cluster_eps, 
                             filter_cluster_size)
        
        pc_filtered = pc[cond]
        if pc_filtered.shape[0] > 0:
            db = sklearn.cluster.DBSCAN(min_samples=min_samples,
                                        eps=eps,
                                        leaf_size=leafSize,
                                        metric="euclidean",
                                        algorithm="ball_tree").fit(pc_filtered)

            labels[cond, 0] = db.labels_

    return np.concatenate([pc, labels], axis=1)

def measure_cluster(cluster):
    nb_cluster = 0
    mean_volume = 0
    total_volume = 0
    if cluster.shape[0] == 0:
        return nb_cluster, mean_volume, total_volume
    labels = cluster[:, 3]
    # Get number of cluster
    nb_cluster = int(max(labels))

    return nb_cluster, mean_volume, total_volume


def pointcloud_clustering(pointcloud_list, dbscan_eps, dbscan_minsample, probThreshold, exp_protocol="lowres", verbose=False, write="None"):
    """
        Pointcloud clustering -- mainly defined for apple clustering 
        :param pointcloud_list: iterator, Iterator that returns pointcloud names and pointcloud array
        :param dbscan_eps: float, Distance to evaluate the neightbourhood
        :param dbscan_minsample: int, Min number of neightbours to take into account
        :param probThreshold: float, Probability threshold of ensure the binary classes 
        :param exp_protocol: str, define the type of protocol to process, options, highres, lowres, synthetic, default: lowres
        :param write: str, Write the point clouds, if None it wont write the point clouds 
        :param verbose: bool, print messages 
        :return:
            iterator (pointcloud_name, pointcloud_nupy_array)
    """
    if exp_protocol == "lowres":
        obj2set = DBSCAN_lowres
        obj2set.eps = DBSCAN_lowres.eps if dbscan_eps < 0 else dbscan_eps
        obj2set.min_samples = DBSCAN_lowres.min_samples if dbscan_minsample < 0 else dbscan_minsample
    elif exp_protocol == "highres":
        obj2set = DBSCAN_highres
        obj2set.eps = DBSCAN_highres.eps if dbscan_eps == -1 else dbscan_eps
        obj2set.min_samples = DBSCAN_highres.min_samples if dbscan_minsample == -1 else dbscan_minsample

    for a_file in pointcloud_list:
        pointcloud_name, pointcloud_array = load_pointcloud(a_file)

        idx2apple = np.where(pointcloud_array[:,3]>=probThreshold)
        idx2zero = np.where(pointcloud_array[:,3]<probThreshold)

        pointcloud_array[idx2apple, 3] = 1
        pointcloud_array[idx2zero, 3] = 0

        pc = clustering(pointcloud_array,
                        10,
                        0.1,
                        None,
                        None)
        
        np.savetxt(pointcloud_name, pc)
        print(np.unique(pc[:,3].astype(np.uint8)))
        print(np.unique(pc[:,4].astype(np.uint8)))
        sys.exit()
        nb_cluster, mean_volume, total_volume = measure_cluster(pc)

        print(" -> Pointcloud name: %s" %(pointcloud_name))
        print(" -> Number of clusters: %i" %(nb_cluster))
        print(" -> Mean volume: %.3f" %(mean_volume))
        print(" -> Total volume: %.3f" %(total_volume))

def main():
    parser = argparse.ArgumentParser("Apple clustering on pointclouds")
    parser.add_argument("path2pointclouds", type=str, help="Path to the segmented pointclouds")
    parser.add_argument("--path2write", type=str, help="Path to write the processed point clouds, default:none", default="none")
    parser.add_argument("--dbscan_eps", type=float, help="eps - Maximum distance between two samples\
                         for one to be considered as in the neighborhood of the other, if -1 use predefined configuration\
                        , default: -1", default=-1)
    parser.add_argument("--dbscan_minsamples", type=int, help="Min samples - The number of samples (or total weight) in a\
                         neighborhood for a point to be considered as a core point. if -1 use default config. default:-1", default=-1)
    parser.add_argument("--cls_prob_threshold", help="Classifier probability threshold, all the values equal of bigger to this variable are\
                         going to set to 1 else to 0, if -1 use default config, default: -1", default=-1)
    parser.add_argument("--higres_protocol", help="Process highresolution point clouds, if not set it assume the pointclouds\
                         were taken using the lowres protocol", action="store_true")
    parser.add_argument("--synthetic", help="Use if you are going to process the synthetic data", action="store_true")
    parser.add_argument("--verbose", help="Print messages showing the advance of the script", action="store_true")
    parser.add_argument("--cores", type=int, help="Number of cores to process the data, default:1", default=1)
    args = parser.parse_args()


    print("-o Pointcloud clustering o- ")
    list_files = glob.glob(os.path.join(args.path2pointclouds, "*.txt"))
    print("  -> Found files: %i" %(len(list_files)))
    if args.cores > 1:
        list_files = np.array_split(list_files, args.cores)

    if args.higres_protocol and not args.synthetic:
        exp = "highres"
    elif args.synthetic and not args.highres_protocol:
        exp = "synthetic"
    else:
        exp = "lowres"

    

    if(args.cores==1):
        clusters = pointcloud_clustering(list_files,
                                        dbscan_eps=args.dbscan_eps,
                                        dbscan_minsample=args.dbscan_minsamples,
                                        probThreshold=args.cls_prob_threshold,
                                        write=args.path2write, 
                                        exp_protocol=exp,
                                        verbose=args.verbose)
    else:
        with Pool(args.cores) as p:
            clusters = p.map(partial(pointcloud_clustering, dbscan_eps=args.dbscan_eps,
                                                            dbscan_minsample=args.dbscan_minsamples,
                                                            probThreshold=args.cls_prob_threshold,
                                                            verbose=args.verbose,
                                                            exp_protocol=exp, 
                                                            write=args.path2write), 
                                                            list_files)

    return 0

if __name__ == "__main__":
    sys.exit(main())