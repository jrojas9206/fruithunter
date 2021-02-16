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
import math
import numpy
import random
import pandas
import os
import pathlib
import sklearn
import sklearn.cluster
import numpy
import scipy.spatial
import matplotlib.pyplot

import openalea.fruithunter.multiprocess
import openalea.fruithunter.filtering
# ==============================================================================


def post_processing_pipeline(filename, save_step=True):

    basename = os.path.basename(filename)
    date, line, pos = basename[3: 13], basename[15:17], basename[19:21]
    pc = numpy.genfromtxt(filename, delimiter=' ')

    # # ========================================================================
    # # Filtering on LIDAR information

    nb_point_filtered = numpy.count_nonzero(pc[:, 3] > 0)

    point_cloud = pc[pc[:, 3] > 0][:, 0:3]

    kdtree = scipy.spatial.KDTree(point_cloud, leafsize=10)
    r = kdtree.query_ball_point(point_cloud, 0.10)
    point_cloud = point_cloud[numpy.array(list(map(len, r))) >= 50]

    db = sklearn.cluster.OPTICS(min_samples=50,
                                max_eps=0.10,
                                metric="euclidean",
                                algorithm="ball_tree").fit(point_cloud)
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    if save_step:
        colors = numpy.zeros((labels.shape[0], 3))
        camp = matplotlib.pyplot.get_cmap('hsv', n_clusters_)

        l = list(range(n_clusters_))
        random.shuffle(l)

        for i, k in enumerate(l):
            colors[labels == k] = numpy.array(camp(i)[0:3])

        a = numpy.column_stack([point_cloud, colors, db.labels_])
        numpy.savetxt('{}_apple_colored.txt'.format(basename[:-4]), a)

    return date, line, pos, 0, nb_point_filtered, n_clusters_


def post_processing_pipeline_2(filename, save_step=True):

    basename = os.path.basename(filename)
    date, line, pos = basename[3: 13], basename[15:17], basename[19:21]
    pc = numpy.genfromtxt(filename, delimiter=' ')

    # # ========================================================================
    # # Filtering on LIDAR information

    # Take just apple
    nb_point_filtered = numpy.count_nonzero(pc[:, 3] > 0)
    point_cloud = pc[pc[:, 3] > 0][:, 0:3]

    mmin = numpy.amin(point_cloud, axis=0)
    mmax = numpy.amax(point_cloud, axis=0)
    print(mmin, mmax)
    print((mmax - mmin).astype(int))
    bbox = numpy.zeros(((mmax - mmin) / 0.01).astype(int))
    print(bbox.shape)

    for x, y, z in ((point_cloud - mmin) * 100).astype(int):
        bbox[x - 5: x + 5, y - 5: y + 5, z - 5:z + 5] = 1

    xx, yy, zz = numpy.where(bbox == 1)
    pts = numpy.column_stack([xx, yy, zz])
    pts = numpy.unique(pts, axis=0) / 100 + mmin

    print(point_cloud.shape, pts.shape, filename)
    numpy.savetxt("test.txt", pts)

    # # ========================================================================

    kdtree = scipy.spatial.KDTree(point_cloud, leafsize=1)

    for i, pt in enumerate(pts):
        intercept = kdtree.query_ball_point(pt, 0.10)
        if len(intercept) > 50:
            r = scipy.spatial.distance.cdist([pt], point_cloud[intercept])
            v = numpy.bitwise_and(0.02 < r, r < 0.04)
            nb = numpy.count_nonzero(v)
            if nb > 50:
                print(i, len(pts), pt)

    # point_cloud = point_cloud[numpy.array(list(map(len, r))) >= 50]
    # scipy.spatial.distance.cdist(
    # print(point_cloud)

    exit()

    kdtree = scipy.spatial.KDTree(point_cloud, leafsize=10)
    r = kdtree.query_ball_point(point_cloud, 0.10)
    point_cloud = point_cloud[numpy.array(list(map(len, r))) >= 50]

    db = sklearn.cluster.OPTICS(min_samples=50,
                                max_eps=0.10,
                                metric="euclidean",
                                algorithm="ball_tree").fit(point_cloud)
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    if save_step:
        colors = numpy.zeros((labels.shape[0], 3))
        camp = matplotlib.pyplot.get_cmap('hsv', n_clusters_)

        l = list(range(n_clusters_))
        random.shuffle(l)

        for i, k in enumerate(l):
            colors[labels == k] = numpy.array(camp(i)[0:3])

        a = numpy.column_stack([point_cloud, colors, db.labels_])
        numpy.savetxt('{}_apple_colored.txt'.format(basename[:-4]), a)

    return date, line, pos, 0, nb_point_filtered, n_clusters_


def post_processing_pipeline_3(filename, save_step=True):

    basename = os.path.basename(filename)
    date, line, pos = basename[3: 13], basename[15:17], basename[19:21]
    pc = numpy.genfromtxt(filename, delimiter=' ')

    # # ========================================================================
    # # Filtering on LIDAR information

    point_cloud = pc[pc[:, 3] > 0][:, 0:3]
    # point_cloud = point_cloud[:1000, :]
    kdtree = scipy.spatial.cKDTree(point_cloud, leafsize=40)
    # intercept = kdtree.query_ball_point(point_cloud, 0.10)

    mask = numpy.ones(point_cloud.shape[0], numpy.bool)

    pts_ok = list()
    radius_min = 0.01
    radius_max = 0.07
    for i, pt in enumerate(point_cloud):

        intercept = kdtree.query_ball_point(pt, radius_max * 2.5)
        if len(intercept) > 50:
            inter = point_cloud[mask][intercept]

            bbox = numpy.amax(inter, axis=0) - numpy.amin(inter, axis=0)
            center = numpy.mean(inter, axis=0)

            x0 = numpy.concatenate([center, [0.05]])

            bounds = [tuple(numpy.concatenate([center - radius_min, [radius_min]])),
                      tuple(numpy.concatenate([center + radius_max, [radius_max]]))]

            def func1(x0):
                return numpy.sum(numpy.abs(numpy.linalg.norm(inter - x0[:3], axis=1) - x0[3]))

            def func2(x0):
                return x0[3] / (1 + numpy.count_nonzero(
                    numpy.abs(numpy.linalg.norm(inter - x0[:3], axis=1) - x0[
                                3]) <= 0.015))

            # v = scipy.optimize.least_squares(
            # 	func2,
            # 	x0,
            # 	bounds=bounds)

            bounds = list(zip(numpy.concatenate([center - radius_min, [radius_min]]),
                              numpy.concatenate([center + radius_max, [radius_max]])))
            v = scipy.optimize.dual_annealing(func2, bounds, maxiter=1000)

            cond = numpy.abs(numpy.linalg.norm(
                inter - v.x[:3], axis=1) - v.x[3]) <= 0.015

            if numpy.count_nonzero(cond) > 20:
                print(i, point_cloud.shape[0])
                # r = numpy.column_stack([inter[cond], numpy.ones(inter[cond].shape[0]) * i])

                pts_ok.append(v.x)
                mask[numpy.where(mask)[0][intercept][cond]] = False
                kdtree = scipy.spatial.cKDTree(point_cloud[mask], leafsize=40)

    # numpy.savetxt("{}_test.txt".format(basename), numpy.concatenate(pts_ok))
    output_dir = "outputdir"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    numpy.savetxt(os.path.join(output_dir, "{}_test.txt".format(basename)),
                  numpy.array(pts_ok))

    print(date, line, pos, 0, 0, len(pts_ok))
    return basename, date, line, pos, 0, 0, len(pts_ok)


def post_processing_custom_ransac(filename, output_dir="output"):

    basename = os.path.basename(filename)
    date, line, pos = basename[3: 13], basename[15:17], basename[19:21]
    print("Loading file :", filename)
    pc = numpy.loadtxt(filename)
    print("End loading file")
    # # ========================================================================
    # # Filtering on LIDAR information

    point_cloud = pc[pc[:, 3] > 0][:, 0:3]

    nb_point = numpy.count_nonzero(pc[:, 3] > 0)
    print("Start Kdtree computing")
    kdtree = scipy.spatial.cKDTree(point_cloud, leafsize=1000)
    print("End Kdtree computing")
    mask = numpy.ones(point_cloud.shape[0], numpy.bool)

    pts_ok, volumes = list(), list()
    radius_min = 0.01
    radius_max = 0.07
    for i, pt in enumerate(point_cloud):
        print(i, len(point_cloud))
        intercept = kdtree.query_ball_point(pt, radius_max)
        print(i, len(point_cloud))
        if len(intercept) > 400:
            inter = point_cloud[mask][intercept]

            best = 0
            best_cond = None
            model = None
            for k in range(10):
                sub_sample = numpy.random.choice(intercept, 100)
                sub_inter = point_cloud[mask][sub_sample]

                # evaluate model
                center = numpy.mean(sub_inter, axis=0)
                x0 = numpy.concatenate([center, [0.05]])
                bounds = [tuple(numpy.concatenate([center - radius_min, [radius_min]])),
                          tuple(numpy.concatenate([center + radius_max, [radius_max]]))]

                v = scipy.optimize.least_squares(
                    lambda x: numpy.mean(
                        numpy.abs(
                            numpy.linalg.norm(sub_inter - x[:3], axis=1) - x[3])),
                    x0,
                    bounds=bounds)

                cond = numpy.abs(numpy.linalg.norm(inter - v.x[:3], axis=1) -
                                 v.x[3]) <= 0.01

                if numpy.count_nonzero(cond) > best:
                    model = v
                    best_cond = cond
                    best = numpy.count_nonzero(cond)

            if best > 100:
                pts_ok.append(model.x)
                volumes.append((4.0 * math.pi * (model.x[3] * 100)**3) / 3.0)
                mask[numpy.where(mask)[0][intercept][best_cond]] = False
                kdtree = scipy.spatial.cKDTree(
                    point_cloud[mask], leafsize=1000)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    numpy.savetxt(os.path.join(output_dir, "{}_test.txt".format(basename)),
                  numpy.array(pts_ok))

    print(basename, date, line, pos, nb_point, len(pts_ok), sum(volumes))
    return basename, date, line, pos, nb_point, len(pts_ok), sum(volumes)


def count_result():

    filenames = glob.glob('pc_2018_08_02_L04_P01*.txt')

    results = list()
    for filename in filenames:
        basename = os.path.basename(filename)
        date, line, pos = basename[3: 13], basename[15:17], basename[19:21]
        pc = numpy.genfromtxt(filename, delimiter=' ')
        results.append((date, line, pos, 0, 0, numpy.count_nonzero(pc)))
    results = numpy.array(results)

    print(results)
    df = pandas.DataFrame(results,
                          columns=['date',
                                   'line',
                                   'position',
                                   'nb_point_filtered',
                                   'nb_point_fruit_detected',
                                   'nb_fruit_detected'])

    # df.to_csv('synth_detection_measurements.csv', index=None)
    df.to_csv('detection_measurements.csv', index=None)


def post_processing_synthetic():

    input_folder = "/home/artzet_s/code/dataset/synthetic_lidar_simulation/"

    filenames = glob.glob(input_folder + '*.txt')

    results = openalea.fruithunter.multiprocess.multiprocess_function(
        post_processing_custom_ransac,
        filenames,
        nb_process=1)

    results = numpy.array(results)

    df = pandas.DataFrame(results,
                          columns=['basename',
                                   'date',
                                   'line',
                                   'position',
                                   'nb_point_fruit_detected',
                                   'nb_fruit_detected',
                                   'volumes'])

    df.to_csv('synthetic_measurements.csv', index=None)


def post_processing_labeled():

    input_folder = "/home/artzet_s/code/dataset/afef_apple_tree_filtred_labeled/labeled/"

    filenames = glob.glob(input_folder + '*.txt')

    results = openalea.fruithunter.multiprocess.multiprocess_function(
        post_processing_custom_ransac,
        filenames,
        nb_process=4)

    results = numpy.array(results)

    df = pandas.DataFrame(results,
                          columns=['basename',
                                   'date',
                                   'line',
                                   'position',
                                   'nb_point_fruit_detected',
                                   'nb_fruit_detected',
                                   'volumes'])

    df.to_csv('labeled_measurements.csv', index=None)


def post_processing_field():

    input_folder = "/home/artzet_s/code/dataset/eval_random_forest/"
    #input_folder = "/home/ubuntu/data/mydatalocal/geometric_feature_pcl/"
    filenames = glob.glob(input_folder + '*.txt')

    results = openalea.fruithunter.multiprocess.multiprocess_function(
        post_processing_custom_ransac,
        filenames,
        nb_process=1)

    results = numpy.array(results)

    df = pandas.DataFrame(results,
                          columns=['basename',
                                   'date',
                                   'line',
                                   'position',
                                   'nb_point_fruit_detected',
                                   'nb_fruit_detected',
                                   'volumes'])

    df.to_csv('field_measurements.csv', index=None)


if __name__ == "__main__":

    # post_processing_synthetic()
    # post_processing_labeled()
    post_processing_field()
