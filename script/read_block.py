import os
import numpy
import sys
import glob

def read_npy(input_filename, output_dir):

	data = numpy.load(input_filename)

	if not os.path.exists(output_dir):
		os.mkdir(output_dir)

	
	basename = os.path.basename(input_filename)[:-4]
	output_filename = os.path.join(output_dir, basename)

	for i in range(data.shape[0]):
		numpy.savetxt("{}_{}.txt".format(output_filename, i),  data[i, :, :])


if __name__ == "__main__":
	print(sys.argv)
	read_npy(sys.argv[1], "output")