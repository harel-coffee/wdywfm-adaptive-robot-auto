import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from analyser import NaiveBayesTypeAnalyser, TunedTransformerTypeAnalyser
from controller import AutonomicManagerController
from environment import PERSONAL_IDENTITY_CLASS, GROUP_IDENTITY_CLASS, CLASS_TO_TYPE, EmergencyEvacuationEnvironment
from synthetic_runner import INTERACTIONS_PER_SCENARIO, NUM_SCENARIOS, run_scenario, SEED


def plot_word_counts(tweets_dataframe):
    words_counts = tweets_dataframe["text"].str.split().apply(len)
    words_counts.hist(bins=np.linspace(0, 100, 100), grid=False, edgecolor="C0")
    plt.title("Words per Tweet")
    plt.xlabel("Words")
    plt.ylabel("Tweets")
    plt.show()


def get_dataset():
    tweets_dataframe = pd.read_csv("data/survivor_responses.csv")
    tweets_dataframe["will_help"] = tweets_dataframe["will_help"].astype(int)
    return tweets_dataframe


def configure_tuned_transformer(tweets_dataframe, test_size):
    type_analyser = TunedTransformerTypeAnalyser()
    _, testing_dataframe = type_analyser.train(tweets_dataframe, test_size)

    text_test_features = None
    label_test_array = None

    return type_analyser, text_test_features, label_test_array


def configure_naive_bayes(tweets_dataframe, test_size):
    tweet_label = tweets_dataframe.pop("will_help")
    tweet_text = tweets_dataframe.pop("text")
    text_train, text_test, label_train, label_test = train_test_split(tweet_text,
                                                                      tweet_label,
                                                                      stratify=tweet_label,
                                                                      test_size=test_size,
                                                                      random_state=SEED)

    type_analyser = NaiveBayesTypeAnalyser()
    type_analyser.train(text_train, label_train)

    raw_text_features = type_analyser.convert_text_to_features(text_test)
    text_test_features = raw_text_features.toarray()
    label_test_array = label_test.to_numpy()

    label_test_predicted = type_analyser.predict_type(text_test_features)
    logging.info(
        classification_report(label_test, label_test_predicted, target_names=[CLASS_TO_TYPE[PERSONAL_IDENTITY_CLASS],
                                                                              CLASS_TO_TYPE[GROUP_IDENTITY_CLASS]]))

    return type_analyser, text_test_features, label_test_array


def main():
    test_size = 0.5
    tweets_dataframe = get_dataset()

    # type_analyser, text_test_features, label_test_array = configure_naive_bayes(tweets_dataframe, test_size)
    type_analyser, text_test_features, label_test_array = configure_tuned_transformer(tweets_dataframe, test_size)

    robot_controller = AutonomicManagerController(type_analyser)
    emergency_environment = EmergencyEvacuationEnvironment(text_test_features, label_test_array,
                                                           INTERACTIONS_PER_SCENARIO)
    _ = run_scenario(robot_controller, emergency_environment, NUM_SCENARIOS)


if __name__ == "__main__":
    np.random.seed(SEED)
    logging.basicConfig(level=logging.INFO)
    main()
