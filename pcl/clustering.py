import os 
import sys 
import glob
import argparse
import numpy as np
import sklearn.cluster 
from post_processing.algorithm import clustering

def main(argv):
    parser = argparse.ArgumentParser("Cluster point cloud")
    parser.add_argument("input", type=str, help="InputFolder")
    parser.add_argument("output", type=str, help="Output folder")
    args = parser.parse_args()

    lst_file = glob.glob( os.path.join( args.input, "*.txt" ) )
    olst_file = glob.glob( os.path.join( args.output, "*.txt" ) )
    olst_file = [ os.path.split( i )[-1] for i in olst_file ]
    print(" -> Found Files: %i" %(len(lst_file)))
    eps, minSamples = 0.1, 17
    for idx, afile in enumerate(lst_file):
        fname = os.path.split( afile )[-1]
        if("high" in fname or fname in olst_file or "2019" in fname ):
            print(" -> skiped: %s" %(fname))
            continue
        print("-> Loading[%i/%i]: %s" %(idx, len(lst_file), fname))
        a_pc = np.loadtxt(afile)
        cluster = clustering(a_pc, minSamples, eps, leafSize=18)
        p2wrt = os.path.join( args.output, fname )
        print("-> saving: %s" %p2wrt)
        np.savetxt(p2wrt, cluster)
    print("EXIT")
    return(0) 


if(__name__=="__main__"):
    sys.exit(main(sys.argv))