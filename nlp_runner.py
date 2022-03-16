import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

from analyser import NaiveBayesTypeAnalyser, TunedTransformerTypeAnalyser
from controller import AutonomicManagerController
from environment import PERSONAL_IDENTITY_CLASS, GROUP_IDENTITY_CLASS, CLASS_TO_TYPE, EmergencyEvacuationEnvironment
from synthetic_runner import INTERACTIONS_PER_SCENARIO, NUM_SCENARIOS, run_scenario, SEED
from imblearn.over_sampling import RandomOverSampler

TEXT_CONTENT_COLUMN = "text"
TEXT_LABEL_COLUMN = "label"
GROUP_IDENTITY_LABEL = "Group"


def plot_word_counts(tweets_dataframe):
    words_counts = tweets_dataframe[TEXT_CONTENT_COLUMN].str.split().apply(len)
    words_counts.hist(bins=np.linspace(0, 100, 100), grid=False, edgecolor="C0")
    plt.title("Words per Tweet")
    plt.xlabel("Words")
    plt.ylabel("Tweets")
    plt.show()


def get_dataset():
    tweets_dataframe = pd.read_csv("data/emergency_interactions.csv")
    logging.info("Before label consolidation")
    logging.info(tweets_dataframe[TEXT_LABEL_COLUMN].value_counts())

    tweets_dataframe[TEXT_LABEL_COLUMN] = tweets_dataframe[TEXT_LABEL_COLUMN].map(
        lambda label: GROUP_IDENTITY_CLASS if label == GROUP_IDENTITY_LABEL else PERSONAL_IDENTITY_CLASS)
    logging.info("After label consolidation")
    logging.info(tweets_dataframe[TEXT_LABEL_COLUMN].value_counts())
    return tweets_dataframe


def configure_tuned_transformer(tweets_dataframe, test_size, column_for_stratify, random_seed, train=True):
    type_analyser = TunedTransformerTypeAnalyser()

    if train:
        type_analyser.train(tweets_dataframe, test_size, column_for_stratify, random_seed)

    testing_dataframe = pd.read_csv(type_analyser.testing_csv_file)
    text_test_features = type_analyser.convert_text_to_features(testing_dataframe[TEXT_CONTENT_COLUMN])
    label_test_array = testing_dataframe[TEXT_LABEL_COLUMN].to_numpy()

    return type_analyser, text_test_features, label_test_array


def upsample_minority_class(text_train, label_train):
    text_train = np.expand_dims(text_train, axis=1)
    over_sampler = RandomOverSampler(random_state=SEED)

    text_train, label_train = over_sampler.fit_resample(text_train, label_train)
    text_train = np.squeeze(text_train)

    logging.info("text_train.shape {}".format(text_train.shape))
    logging.info("label_train.shape {}".format(label_train.shape))

    return text_train, label_train


def configure_naive_bayes(tweets_dataframe, test_size):
    tweet_label = tweets_dataframe.pop(TEXT_LABEL_COLUMN)
    tweet_text = tweets_dataframe.pop(TEXT_CONTENT_COLUMN)

    text_train, text_test, label_train, label_test = train_test_split(tweet_text,
                                                                      tweet_label,
                                                                      stratify=tweet_label,
                                                                      test_size=test_size,
                                                                      random_state=SEED)

    text_train, label_train = upsample_minority_class(text_train, label_train)
    type_analyser = NaiveBayesTypeAnalyser()
    type_analyser.train(text_train, label_train)

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
    tweets_dataframe = get_dataset()

    type_analyser, text_test_features, label_test_array = configure_naive_bayes(tweets_dataframe, test_size)
    # type_analyser, text_test_features, label_test_array = configure_tuned_transformer(tweets_dataframe, test_size,
    #                                                                                   TEXT_LABEL_COLUMN, SEED,
    #                                                                                   train=True)

    robot_controller = AutonomicManagerController(type_analyser)
    emergency_environment = EmergencyEvacuationEnvironment(text_test_features, label_test_array,
                                                           INTERACTIONS_PER_SCENARIO)
    _ = run_scenario(robot_controller, emergency_environment, NUM_SCENARIOS)


if __name__ == "__main__":
    np.random.seed(SEED)
    logging.basicConfig(level=logging.INFO)
    main()
