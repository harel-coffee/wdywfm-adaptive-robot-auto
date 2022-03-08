import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_dataset():
    tweets_dataframe = pd.read_csv("data/survivor_responses.csv")
    words_counts = tweets_dataframe["text"].str.split().apply(len)
    words_counts.hist(bins=np.linspace(0, 100, 100), grid=False, edgecolor="C0")
    plt.title("Words per Tweet")
    plt.xlabel("Words")
    plt.ylabel("Tweets")
    plt.show()


def main():
    sensor_data, person_type = get_dataset()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
