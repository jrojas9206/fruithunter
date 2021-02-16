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

from algorithm import clustering, measure_cluster
from myio import init_log, load_harvest, load_data
from analysis import save_field, field_comparison, save_field_in_one
# ==============================================================================

def opti_post_process(parameters):

    basename, pc, proba, min_samples, eps, filter_cluster_size, filter_cluster_eps = parameters
    date = basename[5:15]
    line = int(basename[17:19])
    position = int(basename[21:23])
    year = basename[5:9]

    pc = clustering(pc,
                    min_samples,
                    eps,
                    filter_cluster_size,
                    filter_cluster_eps)

    nb_cluster, mean_volume, total_volume = measure_cluster(pc)

    result = (basename,
              date,
              line,
              position,
              year,
              nb_cluster,
              mean_volume,
              total_volume,
              proba,
              min_samples,
              eps,
              filter_cluster_size,
              filter_cluster_eps)

    return result


def optimize_post_process(filenames):

    data = load_data(filenames)
    df_harvest = load_harvest()

    # probas = numpy.round(numpy.arange(0.75, 1, 0.05), decimals=2)
    probas = [0.75]
    # list_min_samples = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # epss = numpy.round(numpy.arange(0.01, 0.10, 0.01), decimals=2)
    # epss = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]

    # list_min_samples = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40, 45, 50, 60, 70, 80, 90, 100]
    # epss = [0.0025, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06]


    list_min_samples = [5, 10, 15, 20, 25]
    epss = [0.007, 0.010, 0.015, 0.020, 0.025, 0.030]
    cluster_sizes = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80]
    cluster_epss = [0.00750, 0.01, 0.0150]

    # Opti OPTICS
    list_min_samples = [10, 15, 20, 25]
    epss = [None]
    cluster_sizes = [None]
    cluster_epss = [None]

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
                            basename = os.path.basename(filename)
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
    imap_it = pool.imap(opti_post_process, elements, chunksize=5)

    d = collections.defaultdict(list)

    for x in imap_it:
        basename, date, line, position, year, nb_cluster, mean_volume, total_volume, proba, min_samples, eps, cluster_size, cluster_eps = x
        v = (proba, min_samples, eps, cluster_size, cluster_eps)
        d[v].append(x)

        if len(d[v]) == len(filenames):

            results = d[v]

            df_measurements = pandas.DataFrame(results,
                                               columns=['basename',
                                                        'date',
                                                        'line',
                                                        'position',
                                                        'year',
                                                        'nb_cluster',
                                                        'mean_volume',
                                                        'total_volume',
                                                        'proba',
                                                        'min_samples',
                                                        'eps',
                                                        'cluster_size',
                                                        'cluster_eps'])

            r2, rmse, opti = field_comparison(df_harvest, df_measurements)


            print("SCORE : {} {} {} {} {:06.4f} => {:04.2f} {:06.0f} {:07.3f}".format(
                cluster_size, cluster_eps, proba, min_samples, eps,  r2, rmse, opti), flush=True)


def compute_figure(filenames, output_dir, proba, min_samples, eps, cluster_size, cluster_eps):

    data = load_data(filenames)
    df_harvest = load_harvest()

    # proba = 0.75
    # min_samples = 40
    # eps = 0.04
    # cluster_size = 30
    # cluster_eps = 0.0150

    elements = list()
    for filename in data:
        basename = os.path.basename(filename)

        cond = data[filename][:, 3] >= proba

        pc = data[filename][cond][:, :3]
        print(data[filename].shape, pc.shape)

        elements.append((basename, pc, proba, min_samples,
                         eps, cluster_size, cluster_eps))
        # elements.append((basename, pc, proba, min_samples, eps))


    print("\n\nNumber : {}\n\n".format(
        len(elements) / len(filenames)), flush=True)

    pool = multiprocessing.Pool(20)
    results = pool.map(opti_post_process, elements, chunksize=5)

    # results = list()
    # for i, element in enumerate(elements):
    #     print("{} / {}".format(i, len(elements)))
    #     results.append(opt_post_process(element))

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)


    df_measurements = pandas.DataFrame(results,
                                        columns=['basename',
                                                    'date',
                                                    'line',
                                                    'position',
                                                    'year',
                                                    'nb_cluster',
                                                    'mean_volume',
                                                    'total_volume',
                                                    'proba',
                                                    'min_samples',
                                                    'eps',
                                                    'cluster_size',
                                                    'cluster_eps'])

    df_comparison = pandas.merge(df_harvest,
                                 df_measurements,
                                 left_on=['line', 'position', 'year'],
                                 right_on=['line', 'position', 'year'])


    R2 = field_comparison(df_harvest, df_measurements)

            # print("SCORE", cluster_size,
            #       cluster_eps, proba, min_samples, eps, "  -  ", R2, flush=True)

    print("SCORE", proba, min_samples, eps, "  -  ", R2, flush=True)


    save_field_in_one(df_comparison, output_dir, "Apple Tree")

    # elements = [
    #     (df_comparison,
    #      "Apple Tree:"),
    #     (df_comparison[df_comparison['position'].isin([1, 5, 6, 10, 11, 15, 16, 20])],
    #      "Apple Tree : POS 1&5"),
    #     (df_comparison[df_comparison['position'].isin([2, 4, 7, 9, 12, 14, 17, 19])],
    #      "Apple Tree : POS 2&4"),
    #     (df_comparison[df_comparison['position'].isin([3, 8, 13, 18])],
    #      "Apple Tree : POS 3"),
    #      (df_comparison[df_comparison['position'].isin([2, 4, 7, 9, 12, 14, 17, 19, 3, 8, 13, 18])],
    #      "Apple Tree : POS 2,3,4")]

    # for df, title in elements:
    #     print(title)
        # save_field(df, output_dir, title)


def compute_figure_field_rf_predicted(protocol='LowRes'):

    input_folder = "/gpfswork/rech/wwk/uqr22pt/pred_RF_field/"
    init_log("/gpfswork/rech/wwk/uqr22pt/post_process")

    if protocol == "LowRes":
        filenames = glob.glob(input_folder + '*.txt')
        filenames = [f for f in filenames if "high_quality" not in f]
    else:
        filenames = glob.glob(input_folder + '*high_quality*.txt')
        
    print(*filenames, sep="\n")

    # optimize_post_process(filenames)

    output_dir = "/gpfswork/rech/wwk/uqr22pt/figure_field_rf_prediction_{}/".format(protocol)
    
    if protocol == "LowRes":

        proba = 0.75
        min_samples = 10
        eps = 0.01
        cluster_size = None
        cluster_eps = None

        # proba = 0.75
        # min_samples = 40
        # eps = 0.04
        # cluster_size = 30
        # cluster_eps = 0.015

    else:
        proba = 0.75
        min_samples = 15
        eps = 0.01
        cluster_size = None
        cluster_eps = None


    compute_figure(filenames, output_dir, proba, min_samples, eps, cluster_size, cluster_eps)

def compute_figure_field_randlanet_predicted(protocol='LowRes'):
    input_folder = "/gpfswork/rech/wwk/uqr22pt/pred_RandLA-Net_field/"
    init_log("/gpfswork/rech/wwk/uqr22pt/pp_randlanet")

    if protocol == "LowRes":
        filenames = glob.glob(input_folder + '*.txt')
        filenames = [f for f in filenames if "high_quality" not in f]
    else:
        filenames = glob.glob(input_folder + '*high_quality*.txt')
        
    print(*filenames, sep="\n")

    #optimize_post_process(filenames)

    output_dir = "/gpfswork/rech/wwk/uqr22pt/figure_field_randlanet_prediction_{}/".format(protocol)

    if protocol == "LowRes":

        proba = 0.75
        min_samples = 10
        eps = 0.01
        cluster_size = None
        cluster_eps = None

        # proba = 0.75
        # min_samples = 40
        # eps = 0.04
        # cluster_size = 30
        # cluster_eps = 0.015

    else:
        proba = 0.75
        min_samples = 15
        eps = 0.01
        cluster_size = None
        cluster_eps = None


    compute_figure(filenames, output_dir, proba, min_samples, eps, cluster_size, cluster_eps)

if __name__ == "__main__":
    compute_figure_field_rf_predicted(protocol="LowRes")
    compute_figure_field_rf_predicted(protocol="HiRes")
    compute_figure_field_randlanet_predicted(protocol="LowRes")
    compute_figure_field_randlanet_predicted(protocol="HiRes")
