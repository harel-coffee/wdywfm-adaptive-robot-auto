from keras import models
from keras import layers
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class TypeAnalyser(object):

    def __init__(self, num_features):
        self.network = models.Sequential()

        self.network.add(layers.Dense(units=16, activation="relu", input_shape=(num_features,)))
        self.network.add(layers.Dense(units=16, activation="relu"))
        self.network.add(layers.Dense(units=1, activation="sigmoid"))

        self.network.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

    def train(self, sensor_data, person_type):
        sensor_data_train, sensor_data_test, person_type_train, person_type_test = train_test_split(sensor_data,
                                                                                                    person_type,
                                                                                                    test_size=0.33,
                                                                                                    random_state=0)

        training_history = self.network.fit(sensor_data_train,
                                            person_type_train,
                                            epochs=10,
                                            verbose=1,
                                            batch_size=100,
                                            validation_data=(sensor_data_test, person_type_test))

        return training_history

    def obtain_probabilities(self, sensor_data):
        return self.network.predict(sensor_data)

    def predict_type(self, sensor_data):
        return self.network.predict_classes(sensor_data)


def get_dataset():
    features, target = make_classification(n_samples=10000,
                                           n_features=100,
                                           n_informative=3,
                                           n_redundant=0,
                                           n_classes=2,
                                           weights=[.5, .5],
                                           random_state=0)

    return features, target


def plot_training(training_history, metric):
    training_metric = training_history.history[metric]
    validation_metric = training_history.history["val_" + metric]

    epoch = range(1, len(training_metric) + 1)
    plt.plot(epoch, training_metric, "r--")
    plt.plot(epoch, validation_metric, "b-")

    plt.legend(["Training " + metric, "Validation " + metric])
    plt.xlabel("Epoch")
    plt.ylabel(metric)

    plt.show()


def main():
    np.random.seed(0)

    sensor_data, person_type = get_dataset()
    sensor_data_train, sensor_data_test, person_type_train, person_type_test = train_test_split(sensor_data,
                                                                                                person_type,
                                                                                                test_size=0.33,
                                                                                                random_state=0)
    print "len(sensor_data_train): ", len(sensor_data_train)
    print "len(sensor_data_test): ", len(sensor_data_test)
    _, num_features = sensor_data.shape
    type_analyser = TypeAnalyser(num_features)
    training_history = type_analyser.train(sensor_data_train, person_type_train)

    plot_training(training_history, "acc")
    type_predictions = type_analyser.predict_type(sensor_data_test)
    accuracy = accuracy_score(person_type_test, type_predictions)
    print "accuracy:", accuracy


if __name__ == "__main__":
    main()
