from keras import layers
from keras import models
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


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

    def train(self, text_train, labels_train):
        text_train_counts = self.count_vectorizer.fit_transform(text_train)
        self.classifier.fit(text_train_counts, labels_train)

    def obtain_probabilities(self, text):
        text_counts = self.count_vectorizer.transform(text)
        return self.classifier.predict_proba(text_counts)

    def predict_type(self, text):
        text_counts = self.count_vectorizer.transform(text)
        return self.classifier.predict(text_counts)
