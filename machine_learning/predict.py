import scipy
import os
import collections
import numpy
import glob
import argparse
import multiprocessing
import sklearn
import sklearn.metrics
import sklearn.utils
import sklearn.model_selection
import tensorflow as tf
from tensorflow import keras
import machine_learning.myio
from machine_learning.RFClassifier import RFClassifier
from machine_learning.NNClassifier import NNClassifier
import logging


def synthetic_predict(classifier, filename, output_dir, index_label=3):

    output_filename = os.path.join(output_dir, os.path.basename(filename))

    if os.path.exists(output_filename):
        print(output_filename)
        return None

    data = numpy.loadtxt(filename)

    selector = [i for i in range(data.shape[1]) if i not in [0, 1, 2, index_label]]

    proba = numpy.round(classifier.predict(data[:, selector]),
                        decimals=2)

    res = numpy.column_stack([data[:, :3], proba])
    numpy.savetxt(output_filename, res)

    print(output_filename)


def predict(model, model_filename, input_dir, output_dir, labeled=False, number_of_process=14):

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    classifier = model()
    classifier.load(model_filename)

    index_label = 6
    filenames = glob.glob(os.path.join(input_dir, "*.txt"))
    elements = [(classifier, f, output_dir, index_label) for f in filenames]

    pool = multiprocessing.Pool(number_of_process)
    pool.starmap(synthetic_predict, elements)


def synthetic_predict_rf(model_filename, inputDIR, output):
    #input_dir = "/home/jprb/Documents/data/Final/protocol-HighResolution_sensorResolution-004-set235/features/kfolds/test/"
    #model_filename = "/home/jprb/Documents/data/protocol_high_res_004/output/model_fold_4.sav"
    model = RFClassifier
    #output_dir = "/home/jprb/Documents/data/Final/protocol-HighResolution_sensorResolution-004-set235/features/kfolds/test/predicted_rdf_f4/"

    predict(model, model_filename, inputDIR, output)


def field_predict_rf():
    input_dir = "/gpfswork/rech/wwk/uqr22pt/data_field_FPFH/test/"
    model_filename = "/gpfswork/rech/wwk/uqr22pt/model_RF-field_rad_fpfh/model_all.sav"
    model = RFClassifier
    output_dir = "/gpfswork/rech/wwk/uqr22pt/pred_RF_field"

    predict(model, model_filename, input_dir, output_dir)


if (__name__=="__main__"):
    parser = argparse.ArgumentParser(description='Random forest predicton script')
    parser.add_argument("model", type=str, help="Path to the model to load")
    parser.add_argument("path2data", type=str, help="Path to the data that want to be predicted")
    parser.add_argument("path2write", type=str, help="Path where must be write the model output")
    args = parser.parse_args()
    # field_predict_rf()
    synthetic_predict_rf(args.model, args.path2data, args.path2write)