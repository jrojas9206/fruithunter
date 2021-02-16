import scipy
import os
import collections
import numpy
import glob
import sklearn
import sklearn.metrics
import sklearn.utils
import sklearn.model_selection
import tensorflow as tf
from tensorflow import keras
import myio
from RFClassifier import RFClassifier
from NNClassifier import NNClassifier
import logging
import argparse

def check_best_proba(classifier, folds, output_dir):

    scores = collections.defaultdict(list)
    for i, fold in enumerate(folds):

        test_data, test_label = fold

        model_filename = os.path.join(
            output_dir, "model_fold_{}.sav".format(i + 1))

        # classifier = model()
        classifier.set_train_test_data(
            None, None, test_data, test_label)
        classifier.load(model_filename)
        test_predict = classifier.predict_test()
        for proba in numpy.arange(0.5, 1, 0.01):
            scores[proba].append(
                classifier.evaluate_with_proba(
                    test_label, test_predict, proba))

    for k, v in scores.items():
        sem = numpy.round(
            scipy.stats.sem(numpy.array(v)),
            decimals=2)
        mean_scores = numpy.round(
            numpy.mean(numpy.array(v), axis=0),
            decimals=2)
        logging.info("{} :\t {}".format(k, str(mean_scores.tolist())))


def train_validation(classifier, folds, output_dir, proba=0.50, retrain=False):

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    scores = list()
    for i, fold in enumerate(folds):
        
        train_data = [x for j, (x, y) in enumerate(folds) if j != i]
        train_label = [y for j, (x, y) in enumerate(folds) if j != i]

        train_data = numpy.concatenate(train_data)
        train_label = numpy.concatenate(train_label)

        test_data, test_label = fold # Split all the features and labels again??

        model_filename = os.path.join(
            output_dir, "model_fold_{}.sav".format(i + 1))

        # classifier = model()
        classifier.set_train_test_data(
            train_data, train_label, test_data, test_label)

        if not os.path.exists(model_filename) or retrain:
            classifier.train()
            classifier.save(model_filename)
        else:
            classifier.load(model_filename)

        score = classifier.test(proba=proba)
        scores.append(score)

    sem = numpy.round(
        scipy.stats.sem(numpy.array(scores)),
        decimals=2)
    mean_scores = numpy.round(
        numpy.mean(numpy.array(scores), axis=0),
        decimals=2)

    info = """

    Mean Score with proba {}:

    | MCC       | Macro F1 | Apple F1 - Score |   Balanced Acc   | M IoU
    |:---------:|:--------:|:----------------:|:----------------:| ------ :|
    | {}-({})   | {}-({})  | {}-({})          | {}-({})          | {} - ({})

    """.format(
        proba,
        mean_scores[2], sem[2],
        mean_scores[0], sem[0],
        mean_scores[3], sem[3],
        mean_scores[1], sem[1],
        mean_scores[4], sem[4])

    print(info, flush=True)
    logging.info(info)

    train_on_all_data(classifier, folds, output_dir, retrain=retrain)


def train_on_all_data(classifier, folds, output_dir, retrain=False):

    model_filename = os.path.join(output_dir, "model_all.sav")
    if not os.path.exists(model_filename) or retrain:
        train_data = [x for x, y in folds]
        train_label = [y for x, y in folds]

        train_data = numpy.concatenate(train_data)
        train_label = numpy.concatenate(train_label)
        
        # classifier = model()
        classifier.set_train_test_data(train_data, train_label, None, None)
        classifier.train()
        classifier.save(model_filename)


def init_log(output_dir):

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_filename = os.path.join(output_dir, "learning.log")
    logging.basicConfig(filename=log_filename,
                        filemode='w', level=logging.INFO)


def train_field_rf():

    input_dir = "/gpfswork/rech/wwk/uqr22pt/data_field_FPFH/"
    output_dir = "/gpfswork/rech/wwk/uqr22pt/new_model_RF-field_rad_fpfh/"

    print("\n\nTraining & Validation : {}\n\n".format(output_dir), flush=True)

    init_log(output_dir)
    folds = myio.load_5_fold(input_dir, index_label=6)

    train_validation(
        RFClassifier(), folds, output_dir, retrain=False)
    #check_best_proba(RFClassifier(), folds, output_dir)


def train_field_rf_only_fpfh():
    input_dir = "/gpfswork/rech/wwk/uqr22pt/data_field_FPFH/"
    output_dir = "/gpfswork/rech/wwk/uqr22pt/new_model_RF-field_fpfh/"

    print("\n\nTraining & Validation : {}\n\n".format(output_dir), flush=True)

    init_log(output_dir)
    folds = myio.load_5_fold(input_dir,
                             index_label=6,
                             index_selection=list(range(7, 40)))

    train_validation(
        RFClassifier(), folds, output_dir, retrain=False)
    #check_best_proba(RFClassifier(), folds, output_dir)


def train_field_nn():
    input_dir = "/gpfswork/rech/wwk/uqr22pt/data_field_FPFH/"
    output_dir = "/gpfswork/rech/wwk/uqr22pt/new_model_NN-field_rad_fpfh/"

    print("\n\nTraining & Validation : {}\n\n".format(output_dir), flush=True)

    init_log(output_dir)
    folds = myio.load_5_fold(input_dir, index_label=6)

    classifier = NNClassifier(input_size=36, layer_size=36)
    train_validation(classifier, folds, output_dir, retrain=False)
    #check_best_proba(classifier, folds, output_dir)


def train_field_nn_only_fpfh():
    input_dir = "/gpfswork/rech/wwk/uqr22pt/data_field_FPFH/"
    output_dir = "/gpfswork/rech/wwk/uqr22pt/new_model_NN-field_fpfh/"

    print("\n\nTraining & Validation : {}\n\n".format(output_dir), flush=True)

    init_log(output_dir)
    folds = myio.load_5_fold(input_dir,
                             index_label=6,
                             index_selection=list(range(7, 40)))

    classifier = NNClassifier(input_size=33, layer_size=33)
    train_validation(classifier, folds, output_dir, retrain=False)
    #check_best_proba(classifier, folds, output_dir)


def train_synthetic_rf():
    input_dir = "/gpfswork/rech/wwk/uqr22pt/data_synthetic_HiHiRes_FPFH/"
    output_dir = "/gpfswork/rech/wwk/uqr22pt/model_RF-synthetic_HiHiRes_FPFH/"

    print("\n\nTraining & Validation : {}\n\n".format(output_dir), flush=True)

    init_log(output_dir)
    folds = myio.load_5_fold(input_dir, index_label=3)

    train_validation(
        RFClassifier(), folds, output_dir, retrain=False)
    #check_best_proba(RFClassifier(), folds, output_dir)


def train_synthetic_nn():
    input_dir = "/gpfswork/rech/wwk/uqr22pt/data_synthetic_HiHiRes_FPFH/"
    output_dir = "/gpfswork/rech/wwk/uqr22pt/new_model_NN-synthetic_HiHiRes_FPFH/"

    print("\n\nTraining & Validation : {}\n\n".format(output_dir), flush=True)

    init_log(output_dir)
    folds = myio.load_5_fold(input_dir, index_label=3)

    classifier = NNClassifier(input_size=33, layer_size=33)
    train_validation(classifier, folds, output_dir, retrain=False)
    #check_best_proba(classifier, folds, output_dir)


def measure_data_size():

    input_dir = "/gpfswork/rech/wwk/uqr22pt/synthetic_5-fold_featured/"
    folds = myio.load_5_fold(input_dir, index_label=3)

    # input_dir = "/gpfswork/rech/wwk/uqr22pt/field_5-fold_feature_labeled/"
    # folds = myio.load_field_5_fold(input_dir)

    number_of_apple = 0
    number_of_not_apple = 0
    number_of_point = 0

    print("Measure FIELD")
    for i, fold in enumerate(folds):
        test_data, test_label = fold

        print(test_data.shape)
        print(test_label.shape)

        cond = test_label == 1
        nb = numpy.count_nonzero(cond)
        number_of_apple += nb
        number_of_not_apple += test_label.shape[0] - nb
        number_of_point += test_label.shape[0]

    print("Number of apple", number_of_apple)
    print("Number of not apple", number_of_not_apple)
    print("Number point", number_of_point)

def train_RDF(input_dir, output_dir):
    init_log(output_dir)
    folds = myio.load_5_fold(input_dir, index_label=3)

    train_validation(
        RFClassifier(), folds, output_dir, retrain=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train the Random Forest Model")
    parser.add_argument("inputDir", type=str, help="Path to the data directory")
    parser.add_argument("--outputDir", type=str, help="Directory to write the output of the training", default="./output")
    parser.add_argument("--modelName", type=str, help="Name of the trained model", default="model")
    parser.add_argument("--saveModelFormat", type=str, help="Format to save the trained model", default="sav")
    parser.add_argument("--trainMethod", type=str, help="Train method to use: kfolds,  classic or all_train", default="classic")
    parser.add_argument("--TMC_train", type=str, help="Name of the folder that contain the train data", default="train")
    parser.add_argument("--TMC_test", type=str, help="Name of the folder that contain the test folder", default="test")
    parser.add_argument("--labelColIdx", type=int, help="Column that contain the annotation of the points", default=3)
    parser.add_argument("--probModel", type=float, help="Probabilite applied to evaluate the RDF model", default=0.50)
    parser.add_argument("--featureIndexSelection_start", type=int, help="Initial column position of the features", default=None)
    parser.add_argument("--featureIndexSelection_end", type=int, help="Final column position of the features", default=None)
    parser.add_argument("--removeXYZ", type=bool, help="Remove the XYZ coordinates from the set, This assume that the coordinates are in the positions 0, 1, 2", default=True)
    args = parser.parse_args()
    # Verify that the out directory exist if not created one 
    if(not os.path.isdir(args.outputDir)):
        os.mkdir(args.outputDir)
    # Depeding on the train Method deploy different action to load and organice the data
    if(args.trainMethod=="classic" or args.trainMethod=="all_train"):
        """
        Load the data from two different folder train and test 
        and apply the metrics over the prediction set 
            - IoU
            - F1
            - MMC
            - Balanced Accuracy 
            - Support 
            - Confusion Matrix
        Note: The name of the folders could be changed from the input 
        arguments 
        """
        train_path = os.path.join(args.inputDir, args.TMC_train)
        if(args.trainMethod == "classic"):
            test_path  = os.path.join(args.inputDir, args.TMC_test)
        saveModel_path = os.path.join(args.outputDir, args.modelName+"."+args.saveModelFormat)
        # Load the files on the realted directories -- only txt and npy files are taken in care  
        print("-> Loading train set")
        train_set = myio.load_data(train_path)
        print("  -> Shape: %s" %str(train_set.shape))
        if(args.trainMethod == "classic"):
            print("-> Loading test set")
            test_set  = myio.load_data(test_path)
            print("  -> Shape: %s" %str(test_set.shape))
        # Split the features and the annotated classes
        print("-> Annotated label was defined on the column %i" %args.labelColIdx) 
        y_train = numpy.array([y[args.labelColIdx] for y in train_set]) # Apple is annotated as 1 and other as 0
        if(args.trainMethod == "classic"):
            y_test  = numpy.array([y[args.labelColIdx] for y in test_set])
            test_set  = numpy.delete(test_set, args.labelColIdx, 1)
        # Remove the annotation
        train_set = numpy.delete(train_set, args.labelColIdx, 1)
        if(args.removeXYZ):
            # Remove XYZ
            for _ in range(3):
                train_set = numpy.delete(train_set, 0, 1)
                if(args.trainMethod == "classic"):
                    test_set  = numpy.delete(test_set,  0, 1)
        # Train and evaluate the model
        print("-> Model is being created") 
        model = RFClassifier()
        if(args.trainMethod == "classic"):
            print("-> Data is being prepare for training")
            model.set_train_test_data(train_set, y_train, test_set, y_test)
        else:
            model.set_train_test_data(train_set, y_train, train_set, y_train)
        print("-> Model is going to be trained")
        model.train()
        print("-> Model is going to be saved")
        model.save(saveModel_path)
        print("-> Model is going to be tested")
        score = model.test(proba=args.probModel)
    elif(args.trainMethod == "kfolds"):
        train_RDF(args.inputDir, args.ourputDir)
    else: 
        print("-> Unknown options, verify the trainMethod option")
    print("-> END")        
    #train_synthetic_rf()
    #train_synthetic_nn()

    #train_field_rf()
    #train_field_rf_only_fpfh()

    #train_field_nn()
    #train_field_nn_only_fpfh()
