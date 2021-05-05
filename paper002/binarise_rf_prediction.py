import os 
import sys 
import glob 
import argparse
import numpy as np 

def main(argv):
    lstOfFiles = glob.glob(os.path.join(str(argv[1]), "*.txt" ))
    print("Found files: %i" %( len(lstOfFiles) ))

    for idx, pcFile in enumerate(lstOfFiles):
        print( "-> Loading[%i/%i]" %( idx+1, len(lstOfFiles) ), end="\r" if idx<len(lstOfFiles) else "\n" )
        pc = np.loadtxt( pcFile )
        clss = np.where( pc[:,3] > 0.6, 1, 0)
        pc[:,3] = clss

        np.savetxt(pcFile, pc)

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))