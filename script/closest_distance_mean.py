import scipy
import scipy.spatial
import numpy
import os
import sys


def main(filename):
    basename = os.path.basename(filename)
    date, line, pos = basename[3: 13], basename[15:17], basename[19:21]
    pc = numpy.genfromtxt(filename, delimiter=' ')

    # # ========================================================================
    # # Filtering on LIDAR information

    point_cloud = pc[pc[:, 3] > 0][:, 0:3]
    point_cloud = point_cloud[:1000, :]

    kdtree = scipy.spatial.cKDTree(point_cloud, leafsize=40)

    distances = list()

    for i, pt in enumerate(point_cloud):
        d, i = kdtree.query(pt, k=2)
        distances.append(d[1])

    print(sum(distances) / len(distances))


if __name__ == "__main__":
    main(sys.argv[1])
