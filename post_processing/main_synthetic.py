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
import functools
import random
import time
import pandas
import collections
import os
import pathlib
import multiprocessing
import sklearn
import sklearn.cluster
import numpy
import scipy.spatial
import matplotlib.pyplot
import logging
# ==============================================================================

from algorithm import clustering, measure_cluster
from myio import init_log, load_synthetic, load_data
from analysis import save_synthetic, synthetic_comparison


def opt_post_process(parameters):

    basename, pc, proba, min_samples, eps = parameters

    pc = clustering(pc, min_samples, eps)

    nb_cluster, mean_volume, total_volume = measure_cluster(pc)

    return basename, nb_cluster, mean_volume, total_volume, proba, min_samples, eps


def opt_post_process_2(parameters):

    basename, pc, proba, min_samples, eps, filter_cluster_size, filter_cluster_eps  = parameters

    pc = clustering(pc, min_samples, eps, filter_cluster_size, filter_cluster_eps)

    nb_cluster, mean_volume, total_volume = measure_cluster(pc)

    return basename, nb_cluster, mean_volume, total_volume, proba, min_samples, eps, filter_cluster_size, filter_cluster_eps


def write_instance(parameters):

    filename, src_pc, proba, min_samples, eps = parameters
     
    cond = src_pc[:, 3] >= proba
    pc = src_pc[cond][:, :3]

    pc = clustering(pc, min_samples, eps)
    labels = -1 * numpy.ones((src_pc.shape[0], 1), dtype=numpy.int)
    labels[cond, 0] = pc[:, 3]

    dst_pc = numpy.concatenate([src_pc, labels], axis=1)
    numpy.savetxt(filename, dst_pc)

  


def optimize_post_process(filenames):

    data = load_data(filenames)

    df_synthetic = load_synthetic()

    # probas = numpy.round(numpy.arange(0.50, 1, 0.05), decimals=2)
    # probas = list(probas) + [0.99]
    probas = [0.50]
    list_min_samples = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    epss = [0.01, 0.02, 0.03, 0.04, 0.05]
    
    # probas = [0.50]
    # list_min_samples = [10]
    # epss = [0.03]

    elements = list()

    pc = collections.defaultdict(dict)
    for filename in data:
        for proba in probas:
            cond = data[filename][:, 3] >= proba
            pc[filename][proba] = data[filename][cond][:, :3]

    for proba in probas:
        for min_samples in list_min_samples:
            for eps in epss:
                for filename in data:
                    basename = os.path.basename(filename)[:-4]
                    elements.append(
                        (basename, pc[filename][proba], proba, min_samples, eps))

    print("\n\nNumber : {}\n\n".format(
        len(elements) / len(filenames)), flush=True)

    pool = multiprocessing.Pool(20)
    imap_it = pool.imap(opt_post_process, elements, chunksize=5)
    d = collections.defaultdict(list)

    print("Start : ")
    for x in imap_it:
        basename, nb_cluster, mean_volume, total_volume, proba, min_samples, eps = x

        d[(proba, min_samples, eps)].append(x)

        if len(d[(proba, min_samples, eps)]) == len(filenames):

            results = d[(proba, min_samples, eps)]

            df_measurements = pandas.DataFrame(results,
                                               columns=['basename',
                                                        'nb_cluster',
                                                        'mean_volume',
                                                        'total_volume',
                                                        'proba',
                                                        'min_samples',
                                                        'eps'])

            r2, rmse, opti = synthetic_comparison(df_synthetic, df_measurements)

            print("SCORE : {} {} {:06.4f} => {:04.2f} {:06.0f} {:07.3f}".format(proba, min_samples, eps,  r2, rmse, opti), flush=True)


def optimize_post_process_2(filenames):

    data = load_data(filenames)

    df_synthetic = load_synthetic()

    # probas = numpy.round(numpy.arange(0.50, 1, 0.05), decimals=2)
    # probas = list(probas) + [0.99]
    probas = [0.50]
    list_min_samples = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40, 45, 50, 60, 70, 80, 90, 100]
    epss = [0.0025, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    cluster_sizes = [5, 10, 15, 20, 25, 30, 40, 50, 60]
    cluster_epss = [0.00750, 0.01, 0.0150]
    
    elements = list()

    pc = collections.defaultdict(dict)
    for filename in data:
        for proba in probas:
            cond = data[filename][:, 3] >= proba
            pc[filename][proba] = data[filename][cond][:, :3]

    for cluster_size in cluster_sizes:
        for cluster_eps in cluster_epss:
            for proba in probas:
                for min_samples in list_min_samples:
                    for eps in epss:
                        for filename in data:
                            basename = os.path.basename(filename)[:-4]
                            elements.append(
                                (basename,
                                 pc[filename][proba],
                                 proba,
                                 min_samples,
                                 eps,
                                 cluster_size,
                                 cluster_eps))


    print("\n\nNumber : {} \n\n".format(
        len(elements) / len(filenames)), flush=True)
    pool = multiprocessing.Pool(20)
    imap_it = pool.imap(opt_post_process_2, elements, chunksize=5)

    d = collections.defaultdict(list)

    for x in imap_it:
        basename, nb_cluster, mean_volume, total_volume, proba, min_samples, eps, cluster_size, cluster_eps = x
        v = (proba, min_samples, eps, cluster_size, cluster_eps)
        d[v].append(x)

        if len(d[v]) == len(filenames):

            results = d[v]

            df_measurements = pandas.DataFrame(results,
                                               columns=['basename',
                                                        'nb_cluster',
                                                        'mean_volume',
                                                        'total_volume',
                                                        'proba',
                                                        'min_samples',
                                                        'eps',
                                                        'cluster_size',
                                                        'cluster_eps'])


            r2, rmse, opti = synthetic_comparison(df_synthetic, df_measurements)

            print("SCORE : {} {} {:06.4f} {} {} => {:04.2f} {:06.0f} {:07.3f}".format(
                proba, min_samples, eps,  cluster_size, cluster_eps, r2, rmse, opti), flush=True)


def compute_figure(filenames, output_dir, proba, min_samples, eps):

    data = load_data(filenames)
    df_synthetic = load_synthetic()

    elements = list()
    for filename in data:
        basename = os.path.basename(filename)[:-4]
        basename = basename.split("_pos_")[0]

        cond = data[filename][:, 3] >= proba
        pc = data[filename][cond][:, :3]
        elements.append((basename, pc, proba, min_samples, eps))

    print("\n\nNumber : {}\n\n".format(
        len(elements) / len(filenames)), flush=True)

    pool = multiprocessing.Pool(10)
    results = pool.map(opt_post_process, elements, chunksize=5)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    df_measurements = pandas.DataFrame(
        results,
        columns=['basename',
                 'nb_cluster',
                 'mean_volume',
                 'total_volume',
                 'proba',
                 'min_samples',
                 'eps'])

    r2, rmse, opti = synthetic_comparison(df_synthetic, df_measurements)

    print("SCORE : {} {} {:06.4f} => {:04.2f} {:06.0f} {:07.3f}".format(proba, min_samples, eps,  r2, rmse, opti), flush=True)



    df_comparison = pandas.merge(df_measurements,
                                 df_synthetic)

    save_synthetic(df_comparison, output_dir)


def write_clusterization(filenames, output_dir, proba, min_samples, eps):

    data = load_data(filenames)

    elements = list()
    for filename in data:
        basename = os.path.basename(filename)
        output_filename = os.path.join(output_dir, basename)
        elements.append((output_filename, data[filename], proba, min_samples, eps))

    print("\n\nNumber : {}\n\n".format(
        len(elements) / len(filenames)), flush=True)

    pool = multiprocessing.Pool(10)
    results = pool.map(write_instance, elements, chunksize=5)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)


def compute_figure_synthetic_rf_predicted():
    input_folder = "/gpfswork/rech/wwk/uqr22pt/pred_RF_synthetic_HiHiRes/"
    init_log("/gpfswork/rech/wwk/uqr22pt/post_process")

    filenames = glob.glob(input_folder + '*.txt')

    #optimize_post_process(filenames)

    output_dir = "/gpfswork/rech/wwk/uqr22pt/figure_pred_RF_synthetic_HiHiRes/"
    proba = 0.50
    min_samples = 10
    eps = 0.03
    # compute_figure(filenames, output_dir, proba, min_samples, eps)

    output_dir = "/gpfswork/rech/wwk/uqr22pt/instance_RF_synthetic_HiHiRes/"
    write_clusterization(filenames, output_dir, proba, min_samples, eps)


def compute_figure_synthetic_randlanet_predicted():
    input_folder = "/gpfswork/rech/wwk/uqr22pt/pred_RandLA-Net_synthetic_HiHiRes/"
    init_log("/gpfswork/rech/wwk/uqr22pt/post_process")

    filenames = glob.glob(input_folder + '*.txt')

    # optimize_post_process_2(filenames)

    output_dir = "/gpfswork/rech/wwk/uqr22pt/figure_pred_RandLA-Net_synthetic_HiHiRes/"
    proba = 0.50
    min_samples = 15
    eps = 0.02
    compute_figure(filenames, output_dir, proba, min_samples, eps)

    output_dir = "/gpfswork/rech/wwk/uqr22pt/instance_RandLA-Net_synthetic_HiHiRes/"
    #write_clusterization(filenames, output_dir, proba, min_samples, eps)


def compute_figure_synthetic_gt_predicted(input_folder, output_dir, parameters, opti=False):

    init_log("/gpfswork/rech/wwk/uqr22pt/post_process")

    filenames = glob.glob(input_folder + '*.txt')

    if opti:
        optimize_post_process(filenames)

    proba, min_samples, eps = parameters

    compute_figure(filenames, output_dir, proba, min_samples, eps)


if __name__ == "__main__":
    compute_figure_synthetic_rf_predicted()
    compute_figure_synthetic_randlanet_predicted()

    # Generate figure from synthetic data with perferct prediction

    # input_folder = "/gpfswork/rech/wwk/uqr22pt/data_synthetic_HiHiRes/test/"
    # output_dir = "/gpfswork/rech/wwk/uqr22pt/figure_synthetic_HiHiRes_pred_gt/"
    # proba = 0.50
    # min_samples = 2
    # eps = 0.02
    # compute_figure_synthetic_gt_predicted(input_folder, output_dir, (proba, min_samples, eps), opti=False)

    # input_folder = "/gpfswork/rech/wwk/uqr22pt/data_synthetic_HiRes/test/"
    # output_dir = "/gpfswork/rech/wwk/uqr22pt/figure_synthetic_HiRes_pred_gt/"
    # proba = 0.50
    # min_samples = 2
    # eps = 0.02
    # compute_figure_synthetic_gt_predicted(input_folder, output_dir, (proba, min_samples, eps))

    # input_folder = "/gpfswork/rech/wwk/uqr22pt/data_synthetic_LowRes/test/"
    # output_dir = "/gpfswork/rech/wwk/uqr22pt/figure_synthetic_LowRes_pred_gt/"
    # proba = 0.50
    # min_samples = 2
    # eps = 0.020
    # compute_figure_synthetic_gt_predicted(input_folder, output_dir, (proba, min_samples, eps))
