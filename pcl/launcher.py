import os
import glob
import pathlib
import subprocess
import multiprocessing
import argparse
import sys 

def runit(cmd):
    print(cmd)
    subprocess.run(cmd)

def launch_feature(intput_dir, output_dir):

    regex = os.path.join(intput_dir, "*.txt")

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    elements = list()
    for input_filename in glob.glob(regex):
        output_filename = os.path.join(
            output_dir, os.path.basename(input_filename))

        elements.append(["./build/my_feature",
                         "fpfh",
                         "0.025",  # 2.5cm
                         input_filename,
                         output_filename])

    nb_process = 14
    pool = multiprocessing.Pool(nb_process)
    pool.map(runit, elements)


def launch_segmentation(intput_dir, output_dir):

    regex = os.path.join(intput_dir, "*.txt")

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    elements = list()
    for input_filename in glob.glob(regex):
        output_filename = os.path.join(
            output_dir, os.path.basename(input_filename))

        elements.append(["./build/my_segmentation",
                         "euclidian_clustering",
                         input_filename,
                         output_filename])

    nb_process = 14
    pool = multiprocessing.Pool(nb_process)
    pool.map(runit, elements)

def main():
    parser = argparse.ArgumentParser(description='Tool to get the features and segmenet the tree point clouds')
    parser.add_argument("--action", type=int, help="Action that want to be executed, extract features = 0, segment = 1", default=0)
    parser.add_argument("path2data", type=str, help="Path to the txt files that contain the point clouds")
    parser.add_argument("path2write", type=str, help="Path where you want to write the new data")
    args = parser.parse_args()

    if(args.action==0): # Get the feaures 
        launch_feature(args.path2data,  args.path2write)
    elif(args.action==1): # Segment the Apples 
        launch_segmentation(args.path2data, args.path2write)
    else:
        print("-> Unexpected action value")
        sys.exit(0)

if __name__ == "__main__":
    main()

    #launch_feature("/mnt/afef_apple_tree_filtered/",
    #               "/mnt/afef_apple_tree_filtered_featured")

    #launch_feature("/mnt/5-fold_labeled/fold_1",
    #               "/mnt/5-fold_feature_labeled/fold_1")

    #launch_feature("/mnt/5-fold_labeled/fold_2",
    #               "/mnt/5-fold_feature_labeled/fold_2")

    #launch_feature("/mnt/5-fold_labeled/fold_3",
    #               "/mnt/5-fold_feature_labeled/fold_3")

    #launch_feature("/mnt/5-fold_labeled/fold_4",
    #               "/mnt/5-fold_feature_labeled/fold_4")

    #launch_feature("/mnt/5-fold_labeled/fold_5",
    #               "/mnt/5-fold_feature_labeled/fold_5")

    # launch_segmentation("/mnt/rf_predict_50",
    #                     "/mnt/pp_conditional_euclidian_cluster_50")

    # launch_segmentation("/mnt/rf_predict_model_fold_1_proba_50",
    #                     "/mnt/cec_rf_predict_model_fold_1_proba_50")
