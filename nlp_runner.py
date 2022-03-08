import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from environment import PERSONAL_IDENTITY_CLASS, GROUP_IDENTITY_CLASS, CLASS_TO_TYPE
from analyser import NaiveBayesTypeAnalyser


def plot_word_counts(tweets_dataframe):
    words_counts = tweets_dataframe["text"].str.split().apply(len)
    words_counts.hist(bins=np.linspace(0, 100, 100), grid=False, edgecolor="C0")
    plt.title("Words per Tweet")
    plt.xlabel("Words")
    plt.ylabel("Tweets")
    plt.show()


def get_dataset():
    tweets_dataframe = pd.read_csv("data/survivor_responses.csv")
    tweet_label = tweets_dataframe.pop("will_help").astype(int)
    tweet_text = tweets_dataframe

    return tweet_text["text"], tweet_label


def get_naive_bayes_analyser(text_train, label_train):
    type_analyser = NaiveBayesTypeAnalyser()
    type_analyser.train(text_train, label_train)

    return type_analyser


def main():
    tweet_text, tweet_label = get_dataset()
    text_train, text_test, label_train, label_test = train_test_split(tweet_text,
                                                                      tweet_label,
                                                                      stratify=tweet_label,
                                                                      test_size=0.5,
                                                                      random_state=42)

    naive_bayes_analyser = get_naive_bayes_analyser(text_train, label_train)
    label_test_predicted = naive_bayes_analyser.predict_type(text_test)
    logging.info(
        classification_report(label_test, label_test_predicted, target_names=[CLASS_TO_TYPE[PERSONAL_IDENTITY_CLASS],
                                                                              CLASS_TO_TYPE[GROUP_IDENTITY_CLASS]]))

    print(CLASS_TO_TYPE[naive_bayes_analyser.predict_type(["Sure!"]).item()])
    print(naive_bayes_analyser.obtain_probabilities(["Sure!"]))

    print(CLASS_TO_TYPE[naive_bayes_analyser.predict_type(["Sorry, no!"]).item()])
    print(naive_bayes_analyser.obtain_probabilities(["Sorry, no!"]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
