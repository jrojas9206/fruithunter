import os
import sys
import numpy
import glob


def txt_to_npy(input_dir):

    regex = os.path.join(input_dir, "*.txt")
    for filename in glob.glob(regex):
        data = numpy.loadtxt(filename)
        numpy.save(filename[:-4], data)


def npy_to_txt(input_dir):

    regex = os.path.join(input_dir, "*.npy")
    for filename in glob.glob(regex):
        data = numpy.load(filename)
        numpy.savetxt(filename[-4] + '.txt', data)


if __name__ == "__main__":
    argv = sys.argv

    txt_to_npy(argv[1])
