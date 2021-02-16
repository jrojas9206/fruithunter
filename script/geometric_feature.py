import scipy.spatial
import glob
import os
import math
import numpy
import ray
import time

def compute_geometric_feature(filename, output_dir):
	data = numpy.loadtxt(filename)
	xyz = data[:, :3]
	kdtree = scipy.spatial.cKDTree(xyz, leafsize=100)

	nb_point = data.shape[0]
	buffer = nb_point
	features = numpy.zeros((data.shape[0], 6)) - 1

	for i in range(int(math.ceil(nb_point / buffer))):
		print(filename, i, int(math.ceil(nb_point / buffer)))
		intercept = kdtree.query_ball_point(
			xyz[i * buffer: min((i + 1) * buffer, nb_point), ...], 0.10)

		ind = i * buffer
		for j, m in enumerate(intercept):
			if len(m) > 50:
				v = numpy.linalg.eigvals(numpy.cov(xyz[m].T))
				v.sort()

				indf = ind + j
				features[indf, 0] = (v[2] - v[1]) / v[2]  # Linearity
				features[indf, 1] = (v[1] - v[0]) / v[2]  # Planarity
				features[indf, 2] = v[0] / v[2]  # Sphericity
				features[indf, 3] = (v[2] - v[0]) / v[2]  # Anisotropy
				features[indf, 4] = v[0] + v[1] + v[2]  # Sum
				features[indf, 5] = v[0] / (v[0] + v[1] + v[2])  # Change of curvature

	result = numpy.column_stack([data, features])

	filename = os.path.join(output_dir, os.path.basename(filename)) + '.txt'
	numpy.savetxt(filename, result)


def main():

	input_dir = "/home/artzet_s/code/dataset/afef_apple_tree_filtred"
	output_dir = "/home/artzet_s/code/dataset/geometric_feature"
	if not os.path.exists(output_dir): os.mkdir(output_dir)

	filenames = glob.glob(os.path.join(input_dir, "*.txt"))
	for i, filename in enumerate(filenames):
		compute_geometric_feature(filename, output_dir)


if __name__ == "__main__":
	main()