from keras import layers
from keras import models


class TypeAnalyser(object):

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
