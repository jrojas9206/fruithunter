import os 
import sys 
import glob 
import tarfile
#import request 
import argparse 
import numpy as np 

from sklearn.model_selection import train_test_split
from randlanet.utils.data_prepare_apple_tree import * 
from pcl.launcher import launch_feature

def dataSet(path2files, path2output, model, verbose=False, protocol="field"):
    """
    :INPUT:
        path2files : str of the path to the folder of input files
        path2output: str of the path to the output folder 
        model      : str, "rdf" or "rdnet"
        verbose    : If true print few message of the code steps 
        protocol   : Type of protocol to handle ; synthetic/field/field_only_xyz
    :OUTPUT:
        Write the splitted dataset  on the folder
    """
    # NOTE: This segment will be only executed from the notebook 
    lstOfFiles = glob.glob(os.path.join(path2files,"*.txt"))
    if(verbose):
        print("Found files: %i " %(len(lstOfFiles)))
    # Split the files
    X_train, X_test, _,_ = train_test_split(lstOfFiles, range(len(lstOfFiles)), test_size=0.20, random_state=42)
    if(verbose):
        print(" -> Train set: %i" %len(X_train))
        print(" -> Test set : %i" %len(X_test))
    # Create the directory to keep the test and train sets 
    path2initialSplit = path2output #os.path.join(data2annotatedApples, "dataToRDF")
    if(not os.path.isdir(path2initialSplit)):
        os.mkdir(path2initialSplit)
    lst_folders = []
    if(model=="rdf"):
        lst_folders = ["test", "train"]
    elif(model=="rnet"):
        lst_folders = ["test", "training"]
    else:
        return -1
    for folderName, fileList in zip( lst_folders,[X_test, X_train]):
        path2saveData = os.path.join(path2initialSplit)
        for file2feature in fileList:
            output2wrt = os.path.join(path2saveData, folderName)
            if(not os.path.isdir(output2wrt)):
                os.mkdir(output2wrt)
                print("Folder was created: %s" %output2wrt)
            print("-> Loading: %s" %os.path.split(file2feature)[1])
            file2wrt = os.path.join(output2wrt, os.path.split(file2feature)[1])
            if(model == "rdf"):
                # NOTE: If you change the position or the name of the feature generator change the
                # next string "cmd2feature" [execution command]
                cmd2features = "./../pcl/build/my_feature %s %.3f %s %s" %("fpfh",          # Feature extractor 
                                                                        0.025,           # Grid size 
                                                                        file2feature,    # Input File
                                                                        file2wrt)        # Output File
                print(" -> Running feature extractor")
                os.system(cmd2features)
            else: # RandLA-NET
                if(folderName=="test"):
                    convert_for_test(file2feature, path2saveData, grid_size=0.001, protocol=protocol)
                else:
                    convert_for_training(file2feature, None, path2saveData, grid_size=0.001, protocol=protocol)

def uncompress_tar(strPath2file, folder2unc=None):
    """
    Uncompress a tar file
    :INPUT:
        strPath2file: str, Path to the tar or tar.gz file
        folder2unc: str, If the path is give the data is going to be uncompressed there 
    :OUTPUT:
        integer, if -1 there was one mistake, 0 everything was fine 
    """
    if(not tarfile.is_tarfile(strPath2file)):
        return -1
    tarObj = tarfile.open(strPath2file)
    if(folder2unc is not None):
        tarObj.extractall(folder2unc)
    else:
        tarObj.extractall(folder2unc)

def download_data(path2write):
    """
    Download a file from google drive to the specified folder
    :INPUT:
        path2write: Path to the output folder 
    :OUTPUT:
        integer, 0 if all was fine, -1 if there was an error 
    """
    return 0

def get_fileNameFromPath(strPath):
    """
    From a given file's path ex. /path/to/file.txt 
    split the file's name and their path ex. ['/path/to/', 'file.txt']
    :INPUT:
        strPath: str, Path to the file
    :OUTPUT:
        dict, {path, FileName, format}
    """
    dic2ret = {"path": "", "fileName": "", "format": ""}
    for idx in range(len(strPath)-1, 0, -1):
        if(strPath[idx] == '.'):
            dic2ret["format"] = strPath[idx+1:]
        elif(strPath[idx] == '/'):
            dic2ret["fileName"] = strPath[idx+1:]
            dic2ret["path"] = strPath[0:idx]
            return dic2ret
        else: 
            continue
    return None

def restore_data(strPath):
    """
    Function speicific to restore the data of this repo 
    :INPUT:
        strPath: str, path to the data 
    :OUTPUT:
        This method will create 3 with different features 
        the original: XYZ+Reflectance+Annotatation 
        Cluster:      XYZ+Annotations+Clusters 
        Annotations: XYZ+Annotations 
    """
    lst_files = glob.glob(os.path.join(strPath,"*.txt"))
    path2o = os.path.join(strPath, "original")
    path2c = os.path.join(strPath, "cluster")
    path2a = os.path.join(strPath, "annotations")
    # 
    for path2create in [path2o, path2c, path2a]:
        if(not os.path.isdir(path2create)):
            os.mkdir(path2create)
            print(" -> folder was created: %s" %path2create)
    # Walk ove the list of file neames, download the data, delete the desired columns and save the new files 
    for idx, fname in enumerate(lst_files, start=1):
        print("   -> Loading[%i/%i]: %s" %(len(lst_files), idx, get_fileNameFromPath(fname)["fileName"]))
        # Original files 
        w_pc = np.loadtxt(fname)
        # save the files  
        for path2save, col2rm in zip([path2o, path2c, path2a], [6, [3,4,5], [3,4,5,6]]):
            bck    = np.delete(w_pc, col2rm, axis=1)
            p2save = os.path.join(path2save, get_fileNameFromPath(fname)["fileName"])
            np.savetxt(p2save, bck)
    return 0 

def main(argv):
    parser = argparse.ArgumentParser("Download, uncompress or prepare the data")
    parser.add_argument("--action",   type=str, help="download, uncompress, no action", default="noaction")
    parser.add_argument("--path2tar", type=str, help="Path to tar or tar.gz file", default="../data.tar.xz")
    parser.add_argument("--path2data",type=str, help="Path to the folder with the data", default="../data/merged_xyz_radiometric_Clusters_Annotations/")
    parser.add_argument("--path2out", type=str, help="Path to uncompress the files", default="../data/")
    args = parser.parse_args()
    #
    if(args.action == "uncompress"):
        print("-> Uncompressing")
        uncompress_tar(args.path2tar, folder2unc=args.path2out)
    elif(args.action == "noaction"):
        print("-> No especial action is going to be done")
        pass
    else:
        print("-> Unknown action %s" %(args.action))
    outFpath = os.path.join(args.path2out, "merged_xyz_radiometric_Clusters_Annotations/")
    # Restore the data 
    restore_data(outFpath if args.action == "uncompress" else args.path2data)
    return 0

if(__name__=="__main__"):
    sys.exit(main(sys.argv))