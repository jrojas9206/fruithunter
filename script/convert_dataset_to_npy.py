# -*- python -*-
#
#       Copyright 2019 SIMON ARTZET
#
#       Distributed under the Cecill-C License.
#       See accompanying file LICENSE.txt or copy at
#           http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html
#
# ==============================================================================
import glob
import numpy
import os

import openalea.fruithunter.multiprocess
import openalea.fruithunter.filtering
# ==============================================================================


def convert_fake_labeled_pipeline(filename, save_step=True):

	basename = os.path.basename(filename)
	# date, line, pos = basename[3: 13], basename[15:17], basename[19:21]

	pc = numpy.genfromtxt(filename, delimiter=' ')
	pc_fake_labeled = numpy.column_stack((pc[:, :3],
	                                      numpy.zeros((pc.shape[0], 4))))

	output_dir = "/home/artzet_s/code/dataset/afef_npy"
	output_filename = '{}/{}_fake_labeled'.format(output_dir, basename[:-4])

	numpy.save(output_filename, pc_fake_labeled)


def main():

	# input_folder = "/home/ubuntu/sa_volume/afef_apple_tree/"
	input_folder = "/home/artzet_s/code/dataset/afef_apple_tree/"

	filenames = glob.glob(input_folder + 'pc_2018_??_??_L??_P??.txt')

	openalea.fruithunter.multiprocess.multiprocess_function(
		convert_fake_labeled_pipeline,
		filenames,
		nb_process=4)


if __name__ == "__main__":
	main()
