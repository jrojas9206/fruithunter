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
import sklearn.model_selection
import sklearn.utils
import sklearn.metrics

from ModelClassifier import ModelClassifier


class RFClassifier(ModelClassifier):

    def __init__(self):
        super().__init__()

    def train(self):

        self._model = sklearn.ensemble.RandomForestClassifier(
            n_jobs=14,
            n_estimators=300,
            max_depth=15,
            bootstrap=True,
            class_weight="balanced")

        print(self._train_data.shape)
        print(self._train_label.shape)
        self._model.fit(self._train_data, self._train_label)

    def predict(self, data):
        return self._model.predict_proba(data)[:, 1]

    def save(self, filename):
        pickle.dump(self._model, open(filename, 'wb'))

    def load(self, filename):
        self._model = pickle.load(open(filename, 'rb'))
