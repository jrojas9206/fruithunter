import os 
import sys 
import glob 
import tarfile
#import request 
import argparse 
import numpy as np 

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
    parser.add_argument("--path2tar", type=str, help="Path to tar or tar.gz file", default="data/data.tar.xz")
    parser.add_argument("--path2data",type=str, help="Path to the folder with the data", default="data/merged_xyz_radiometric_Clusters_Annotations/")
    parser.add_argument("--path2out", type=str, help="Path to uncompress the files", default="data/")
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