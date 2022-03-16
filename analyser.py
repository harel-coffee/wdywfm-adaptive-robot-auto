import logging
import subprocess
from subprocess import call

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from keras import layers
from keras import models
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import resample

from environment import GROUP_IDENTITY_CLASS


class SyntheticTypeAnalyser(object):

    def __init__(self, num_features):
        self.network = models.Sequential()

        self.network.add(layers.Dense(units=16, activation="relu", input_shape=(num_features,)))
        self.network.add(layers.Dense(units=16, activation="relu"))
        self.network.add(layers.Dense(units=1, activation="sigmoid"))

        self.network.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

    def train(self, sensor_data, person_type, epochs, batch_size, callbacks=None):
        training_history = self.network.fit(sensor_data,
                                            person_type,
                                            epochs=epochs,
                                            verbose=1,
                                            callbacks=callbacks,
                                            batch_size=batch_size,
                                            validation_split=0.33)

        return training_history

    def obtain_probabilities(self, sensor_data):
        return self.network.predict(sensor_data)

    def predict_type(self, sensor_data):
        return self.network.predict_classes(sensor_data)


class NaiveBayesTypeAnalyser(object):

    def __init__(self):
        self.count_vectorizer = CountVectorizer()
        self.classifier = MultinomialNB()

    def train(self, text_train, labels_train, random_state):
        text_train, labels_train = self.upsample_minority(text_train, labels_train, random_state)
        text_train_counts = self.count_vectorizer.fit_transform(text_train)
        self.classifier.fit(text_train_counts, labels_train)

    def obtain_probabilities(self, text_features):
        return self.classifier.predict_proba(text_features)[:, GROUP_IDENTITY_CLASS]

    def predict_type(self, text_features):
        return self.classifier.predict(text_features)

    def convert_text_to_features(self, text):
        return self.count_vectorizer.transform(text)

    @staticmethod
    def upsample_minority(text_train, labels_train, random_state):
        logging.info("Before upsampling -> text_train.shape {}".format(text_train.shape))
        logging.info("Before upsampling -> label_train.shape {}".format(labels_train.shape))

        text_train = np.expand_dims(text_train, axis=1)
        over_sampler = RandomOverSampler(random_state=random_state)

        text_train, label_train = over_sampler.fit_resample(text_train, labels_train)
        text_train = np.squeeze(text_train)

        logging.info("After upsampling -> text_train.shape {}".format(text_train.shape))
        logging.info("After upsampling -> label_train.shape {}".format(label_train.shape))

        return text_train, label_train


class TunedTransformerTypeAnalyser(object):

    def __init__(self):
        self.training_csv_file = "training_data.csv"
        self.testing_csv_file = "testing_data.csv"

        self.prefix = 'conda run -n p36-wdywfm-adaptive-robot '
        self.python_script = '../transformer-type-estimator/transformer_analyser.py'
        self.training_command = self.prefix + 'python {} --trainlocal --train_csv "{}" --test_csv "{}"'
        self.prediction_command = self.prefix + 'python {} --predlocal --input_text "{}"'

    def train(self, original_dataframe, test_size, label_column, random_seed):
        logging.info("Test size {}".format(test_size))

        training_dataframe, testing_dataframe = train_test_split(original_dataframe,
                                                                 stratify=original_dataframe[label_column],
                                                                 test_size=test_size,
                                                                 random_state=random_seed)

        training_dataframe = self.upsample_minority(training_dataframe, label_column, random_seed)
        training_dataframe.to_csv(self.training_csv_file, index=False)
        logging.info("Training data file created at {}".format(self.training_csv_file))
        testing_dataframe.to_csv(self.testing_csv_file, index=False)
        logging.info("Testing data file created at {}".format(self.testing_csv_file))

        command = self.training_command.format(self.python_script, self.training_csv_file, self.testing_csv_file)
        logging.info("Running {}".format(command))
        exit_code = call(command, shell=True)
        logging.info("exit_code {}".format(exit_code))

    def obtain_probabilities(self, text_features):
        text_as_string = text_features.item()
        command = self.prediction_command.format(self.python_script, text_as_string)

        logging.info("Running {}".format(command))
        standard_output = subprocess.check_output(command, shell=True)
        logging.debug("standard_output {}".format(standard_output))

        return np.array([float(standard_output)])

    @staticmethod
    def convert_text_to_features(text_series):
        return np.expand_dims(text_series.to_numpy(), axis=1)

    @staticmethod
    def upsample_minority(training_dataframe, label_column, random_state):
        logging.info("Counts before upsampling")
        logging.info(training_dataframe[label_column].value_counts())

        group_identity_mask = training_dataframe[label_column] == GROUP_IDENTITY_CLASS
        group_identity_dataframe = training_dataframe[group_identity_mask]
        personal_identity_dataframe = training_dataframe[~group_identity_mask]

        personal_identity_upsample = resample(personal_identity_dataframe,
                                              replace=True,
                                              n_samples=len(group_identity_dataframe),
                                              random_state=random_state)

        training_dataframe = pd.concat([group_identity_dataframe, personal_identity_upsample])

        logging.info("Counts after upsampling")
        logging.info(training_dataframe[label_column].value_counts())
        return training_dataframe
