import glob
import os
import sys 
from shutil import copyfile
import argparse

def copy(start_ind, end_ind, d, output_dir):

    for i in range(start_ind, end_ind):
        basename = "synthetic_{}".format(i)
        if basename in d:
            output_filename = os.path.join(
                output_dir, "{}.txt".format(basename))
            
            copyfile(d[basename], output_filename)

            print("cp {} {}".format(d[basename], output_filename))

def create_k_fold_dir(k, output_dir, d):
    
    fold_dir = os.path.join(output_dir, "fold_{}".format(k))
    if not os.path.exists(fold_dir):
        os.mkdir(fold_dir)

    start_ind = (k - 1) * 20 
    copy(start_ind, start_ind + 20, d, fold_dir)

def create_train_test_dir(input_dir, output_dir):

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    filenames = glob.glob(os.path.join(input_dir, "*.txt"))

    print("Length filenames : {}".format(len(filenames)))
    d = dict()
    for filename in filenames:
        
        basename = os.path.basename(filename)
        if "pos" in basename:
            basename = basename.split("_pos")[0]
        else:
            basename = basename[:-4]
        d[basename] = filename

    for k in range(1, 6):
        create_k_fold_dir(k, output_dir, d)

    test_dir = os.path.join(output_dir, "test")
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    copy(100, 300, d, test_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare different kfolds for the training and test of the ML/DL models')
    parser.add_argument("path2data", type=str, help="Path to the point clouds *.txt files")
    parser.add_argument("path2write", type=str, help="Path to the folder where musr be wrote the data")
    parser.add_argument("--default", type=bool, help="Use the deafult paths -- Jean Zay server", default=False)
    args = parser.parse_args()
    if(not args.default):
        create_train_test_dir(args.path2data, args.path2write)
    else:
        input_dir = "/gpfswork/rech/wwk/uqr22pt/raw_synthetic_HiHiRes"
        output_dir = "/gpfswork/rech/wwk/uqr22pt/data_synthetic_HiHiRes"
        create_train_test_dir(input_dir, output_dir)
        input_dir = "/gpfswork/rech/wwk/uqr22pt/raw_synthetic_HiHiRes_fpfh/"
        output_dir = "/gpfswork/rech/wwk/uqr22pt/data_synthetic_HiHiRes_FPFH"
        create_train_test_dir(input_dir, output_dir)
        input_dir = "/gpfswork/rech/wwk/uqr22pt/raw_synthetic_HiRes"
        output_dir = "/gpfswork/rech/wwk/uqr22pt/data_synthetic_HiRes"
        create_train_test_dir(input_dir, output_dir)    
        input_dir = "/gpfswork/rech/wwk/uqr22pt/raw_synthetic_LowRes"
        output_dir = "/gpfswork/rech/wwk/uqr22pt/data_synthetic_LowRes"
        create_train_test_dir(input_dir, output_dir)

