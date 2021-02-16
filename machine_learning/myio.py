import glob
import os
import multiprocessing
import pickle
import numpy
import scipy.stats
import sklearn
import sklearn.neighbors
import sklearn.ensemble
import sklearn.neural_network
import logging


def banlanced_data(X, Y):

    X_Apple = X[Y == 1]
    Y_Apple = Y[Y == 1]
    X_Noise = X[Y == 0]
    Y_Noise = Y[Y == 0]

    ind = numpy.arange(X_Noise.shape[0])
    numpy.random.shuffle(ind)

    X_Noise = X_Noise[ind][:X_Apple.shape[0], ...]
    Y_Noise = Y_Noise[ind][:Y_Apple.shape[0], ...]

    X = numpy.concatenate([X_Apple, X_Noise])
    Y = numpy.concatenate([Y_Apple, Y_Noise])

    return X, Y


def load_data(input_dir):

    data = list()
    filenames = glob.glob(os.path.join(input_dir, "*.txt"))

    for i, filename in enumerate(filenames):
        logging.info("Load : {}".format(filename))

        npy_filename = filename[:-3] + 'npy'
        if os.path.exists(npy_filename):
            pc = numpy.load(npy_filename)
        else:
            pc = numpy.loadtxt(filename)
            numpy.save(npy_filename, pc)

        data.append(pc)

    data = numpy.concatenate(data)

    logging.info("data size : {}".format(data.shape))

    return data


def load_5_fold(input_dir, index_label=3, index_selection=None):

    folds = list()
    for i in range(1, 6):
        fold_dir = os.path.join(input_dir, "fold_{}".format(i))

        data = load_data(fold_dir)

        if index_selection is None:
            index_selection = [i for i in range(data.shape[1]) if i not in [0, 1, 2, index_label]]

        x_data = data[:, index_selection]
        y_data = data[:, index_label]

        logging.info("data size : {}".format(x_data.shape))
        logging.info("label size : {}".format(y_data.shape))

        folds.append((x_data, y_data))

    return folds


