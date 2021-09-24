from keras import layers
from keras import models
from sklearn.model_selection import train_test_split


class TypeAnalyser(object):

    def __init__(self, num_features):
        self.network = models.Sequential()

        self.network.add(layers.Dense(units=16, activation="relu", input_shape=(num_features,)))
        self.network.add(layers.Dense(units=16, activation="relu"))
        self.network.add(layers.Dense(units=1, activation="sigmoid"))

        self.network.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

    def train(self, sensor_data, person_type, epochs, batch_size):
        sensor_data_train, sensor_data_test, person_type_train, person_type_test = train_test_split(sensor_data,
                                                                                                    person_type,
                                                                                                    test_size=0.33,
                                                                                                    random_state=0)

        training_history = self.network.fit(sensor_data_train,
                                            person_type_train,
                                            epochs=epochs,
                                            verbose=1,
                                            batch_size=batch_size,
                                            validation_data=(sensor_data_test, person_type_test))

        return training_history

    def obtain_probabilities(self, sensor_data):
        return self.network.predict(sensor_data)

    def predict_type(self, sensor_data):
        return self.network.predict_classes(sensor_data)
