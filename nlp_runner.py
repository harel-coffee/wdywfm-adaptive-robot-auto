SEED = 42
import numpy as np

np.random.seed(SEED)
import tensorflow as tf

tf.compat.v1.set_random_seed(SEED)

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

from analyser import NaiveBayesTypeAnalyser, TunedTransformerTypeAnalyser
from controller import AutonomicManagerController, ProSocialRobotController, ProSelfRobotController
from environment import PERSONAL_IDENTITY_CLASS, GROUP_IDENTITY_CLASS, CLASS_TO_TYPE, EmergencyEvacuationEnvironment
from synthetic_runner import run_scenario

TEXT_CONTENT_COLUMN = "text"
TEXT_LABEL_COLUMN = "label"
GROUP_IDENTITY_LABEL = "Group"


def plot_word_counts(dataframe):
    words_counts = dataframe[TEXT_CONTENT_COLUMN].str.split().apply(len)
    words_counts.hist(bins=np.linspace(0, 100, 100), grid=False, edgecolor="C0")
    plt.title("Words per Tweet")
    plt.xlabel("Words")
    plt.ylabel("Tweets")
    plt.show()


def get_dataset():
    dataframe = pd.read_csv("data/emergency_interactions.csv")
    logging.info("Before label consolidation")
    logging.info(dataframe[TEXT_LABEL_COLUMN].value_counts())

    dataframe[TEXT_LABEL_COLUMN] = dataframe[TEXT_LABEL_COLUMN].map(
        lambda label: GROUP_IDENTITY_CLASS if label == GROUP_IDENTITY_LABEL else PERSONAL_IDENTITY_CLASS)
    logging.info("After label consolidation")
    logging.info(dataframe[TEXT_LABEL_COLUMN].value_counts())
    return dataframe


def configure_tuned_transformer(dataframe, test_size, column_for_stratify, random_seed, train=True):
    type_analyser = TunedTransformerTypeAnalyser()

    if train:
        type_analyser.train(dataframe, test_size, column_for_stratify, random_seed)

    testing_dataframe = pd.read_csv(type_analyser.testing_csv_file)
    text_test_features = type_analyser.convert_text_to_features(testing_dataframe[TEXT_CONTENT_COLUMN])
    label_test_array = testing_dataframe[TEXT_LABEL_COLUMN].to_numpy()

    return type_analyser, text_test_features, label_test_array


def configure_naive_bayes(dataframe, test_size):
    tweet_label = dataframe.pop(TEXT_LABEL_COLUMN)
    tweet_text = dataframe.pop(TEXT_CONTENT_COLUMN)

    text_train, text_test, label_train, label_test = train_test_split(tweet_text,
                                                                      tweet_label,
                                                                      stratify=tweet_label,
                                                                      test_size=test_size,
                                                                      random_state=SEED)
    type_analyser = NaiveBayesTypeAnalyser()
    type_analyser.train(text_train, label_train, SEED)

    raw_text_features = type_analyser.convert_text_to_features(text_test)
    text_test_features = raw_text_features.toarray()
    label_test_array = label_test.to_numpy()

    label_test_predicted = type_analyser.predict_type(text_test_features)
    logging.info(
        classification_report(label_test, label_test_predicted, target_names=[CLASS_TO_TYPE[PERSONAL_IDENTITY_CLASS],
                                                                              CLASS_TO_TYPE[GROUP_IDENTITY_CLASS]]))
    logging.info("accuracy_score: {}".format(accuracy_score(label_test, label_test_predicted)))
    return type_analyser, text_test_features, label_test_array


def main():
    test_size = 0.5
    dataframe = get_dataset()

    type_analyser, text_test_features, label_test_array = configure_naive_bayes(dataframe, test_size)
    # type_analyser, text_test_features, label_test_array = configure_tuned_transformer(dataframe, test_size,
    #                                                                                   TEXT_LABEL_COLUMN, SEED,
    #                                                                                   train=True)

    # robot_controller = AutonomicManagerController(type_analyser)
    # robot_controller = ProSocialRobotController()
    robot_controller = ProSelfRobotController()

    emergency_environment = EmergencyEvacuationEnvironment(text_test_features, label_test_array,
                                                           len(label_test_array))
    _ = run_scenario(robot_controller, emergency_environment, 1)


if __name__ == "__main__":
    np.random.seed(SEED)
    logging.basicConfig(level=logging.INFO)
    main()
