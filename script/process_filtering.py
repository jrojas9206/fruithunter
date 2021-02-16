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
import pandas
import os
import pathlib
import sklearn
import sklearn.cluster
import numpy
import matplotlib.pyplot
import scipy.spatial
import multiprocessing


import openalea.fruithunter.multiprocess
import openalea.fruithunter.filtering
# ==============================================================================


def baseline_pipeline(filename):

    basename = os.path.basename(filename)
    date, line, pos = basename[3: 13], basename[15:17], basename[19:21]
    pc = numpy.genfromtxt(filename, delimiter=' ')

    # ==========================================================================

    ind1 = numpy.bitwise_and(-14 <= pc[:, 5], pc[:, 5] <= -9)
    ind2 = numpy.bitwise_and(pc[:, 4] >= 0, pc[:, 4] <= 5)
    ind3 = numpy.bitwise_and(ind1, ind2)
    pc = pc[ind3]

    # numpy.savetxt('{}_filtered.txt'.format(basename[:-4]), pc, delimiter=' ')

    # ==========================================================================

    kdtree = scipy.spatial.cKDTree(pc[:, :3], leafsize=1000)
    r = kdtree.query_ball_point(pc[:, :3], 0.05)  # 0.1 = 10cm
    pc = pc[numpy.array(list(map(len, r))) >= 100]

    # numpy.savetxt('{}_ball_density.txt'.format(basename[:-4]), pc,
    #               delimiter=' ')
    # ==========================================================================

    db = sklearn.cluster.OPTICS(min_samples=100, max_eps=1.).fit(pc[:, :3])
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    # ========================================================================
    # Filtering on geometric feature

    colors = numpy.zeros((labels.shape[0], 3))
    camp = matplotlib.pyplot.get_cmap('tab20', n_clusters_)

    for k in range(n_clusters_):
        colors[labels == k] = numpy.array(camp(k)[0:3])

    a = numpy.column_stack([pc[:, :3], labels, colors])
    numpy.savetxt('{}_apple_colored.txt'.format(basename[:-4]), a)

    print(basename, date, line, pos, n_clusters_)
    return basename, date, line, pos, n_clusters_


def filtering(input_filename, output_filename):

    print(input_filename)
    with open(output_filename, 'w') as outout:
        for line in open(input_filename):
            values = line.split(' ')
            if float(values[7]) <= 15 and int(values[12]) == 1:
                out = "{} {} {} {} {} {}\n".format(values[2],
                                                   values[3],
                                                   values[4],
                                                   values[5],
                                                   values[7],
                                                   values[11])

                outout.write(out)
        outout.close()
    print(output_filename)


def filtering_2018_08_02(input_filename, output_filename):

    print(input_filename)
    with open(output_filename, 'a') as outout:
        for line in open(input_filename):
            if float(line.split(' ')[4]) <= 15:
                outout.write(line)
    print(output_filename)


def main():

    # input_folder = "/home/ubuntu/sa_volume/afef_apple_tree/"
    input_folder = "/home/artzet_s/code/dataset/afef_apple_tree_dataset/trees"
    output_folder = "/home/artzet_s/code/dataset/afef_apple_tree_filtered/"

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    elements = list()
    # filenames = glob.glob(
    #     os.path.join(input_folder, 'tree_2018_08_03_L??_P??.txt')) + glob.glob(
    #         os.path.join(input_folder, 'tree_2019_??_??_L??_P??.txt'))

    filenames = glob.glob(os.path.join(input_folder, 'tree_2019_*high*.txt'))
    print(*filenames, sep="\n")
    for i, input_filename in enumerate(filenames):
        output_filename = os.path.join(
            output_folder, os.path.basename(input_filename))
        elements.append((input_filename, output_filename))

    print(elements)
    pool = multiprocessing.Pool(4)
    pool.starmap(filtering, elements)


if __name__ == "__main__":
    main()
