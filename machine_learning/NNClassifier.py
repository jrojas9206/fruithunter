import logging
import glob
import os
import multiprocessing
import pickle
import numpy
import scipy.stats

from tensorflow import keras
import tensorflow as tf

import sklearn
import sklearn.neighbors
import sklearn.ensemble
import sklearn.neural_network
import sklearn.model_selection
import sklearn.utils
import sklearn.metrics


from ModelClassifier import ModelClassifier


class NNClassifier(ModelClassifier):

    def __init__(self, input_size=33, layer_size=33):

        super().__init__()

        self._model = keras.Sequential([
            keras.layers.Dense(layer_size, activation='relu', input_shape=(input_size,)),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(layer_size, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(layer_size, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(1, activation='sigmoid')
        ])


    def train(self):


        self._model.compile(optimizer='adam',
                            loss="binary_crossentropy",
                            metrics=['accuracy'])

        sk_class_weight = sklearn.utils.class_weight.compute_class_weight(
            'balanced',
            numpy.unique(self._train_label),
            self._train_label)

        logging.info("CLASS WEIGHT: {}".format(sk_class_weight))

        class_weight = {0: sk_class_weight[0],
                        1: sk_class_weight[1]}

        validation_data = None
        if self._test_data is not None:
            validation_data = (self._test_data, self._test_label)

        self._model.fit(self._train_data, self._train_label,
                        epochs=150,
                        batch_size=1000000,
                        validation_data=validation_data,
                        class_weight=class_weight)

    def save(self, filename):
        self._model.save(filename)

    def load(self, filename):
        self._model = tf.keras.models.load_model(filename)

    def predict(self, data):
        return self._model.predict(data, batch_size=100000)


