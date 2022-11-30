import datetime
import logging
import pickle

import numpy as np
import tensorflow as tf
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, History
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from typing import Tuple, Optional, List

from analyser import SyntheticTypeAnalyser, TYPE_TO_CLASS
from controller import AutonomicManagerController
from gamemodel import PERSONAL_IDENTITY_TYPE, SHARED_IDENTITY_TYPE

SEED = 0  # type:int
np.random.seed(SEED)
tf.set_random_seed(SEED)

NUM_SCENARIOS = 10  # type:int
INTERACTIONS_PER_SCENARIO = 10  # type:int

MODEL_FILE = "model/trained_model.h5"  # type:str
ENCODER_FILE = "model/encoder.pickle"  # type:str


class EarlyStoppingByTarget(Callback):

    def __init__(self, monitor='val_acc', target=0.8, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.target = target
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current_value = logs.get(self.monitor)
        if current_value is None:
            logging.error("Early stopping requires %s available!" % self.monitor)
            raise RuntimeWarning()

        if current_value >= self.target:
            if self.verbose > 0:
                logging.info("Epoch %s: early stopping, target accuracy reached" % str(epoch))
            self.model.stop_training = True


def get_log_directory(base_directory="logs/"):
    # type: (str) -> str
    return base_directory + datetime.datetime.now().strftime("run_%Y_%m_%d_%H_%M_%S")


def get_synthetic_dataset(selfish_type_weight, zeroresponder_type_weight, total_samples):
    features, target = make_classification(n_samples=total_samples,
                                           n_features=100,
                                           n_informative=3,
                                           n_redundant=0,
                                           n_classes=2,
                                           weights=[selfish_type_weight, zeroresponder_type_weight],
                                           random_state=0)

    return features, target


def plot_training(training_history, metric, plot_validation=True):
    training_metric = training_history.history[metric]
    legend = ["Training " + metric]  # type: List[str]
    epoch = range(1, len(training_metric) + 1)  # type: List[int]

    plt.plot(epoch, training_metric, "r--")

    if plot_validation:
        validation_metric = training_history.history["val_" + metric]
        plt.plot(epoch, validation_metric, "b-")
        legend.append("Validation " + metric)

    plt.legend(legend)
    plt.xlabel("Epoch")
    plt.ylabel(metric)

    plt.show()


def under_sample(first_index, second_index, original_features, original_classes):
    # type: (np.ndarray, np.ndarray, np.ndarray, np.ndarray) -> Tuple[np.ndarray,np.ndarray]

    majority_index = first_index  # type: np.ndarray
    minority_index = second_index  # type: np.ndarray

    if len(majority_index) < len(minority_index):
        majority_index, minority_index = minority_index, majority_index

    majority_sample_index = np.random.choice(majority_index, size=len(minority_index),
                                             replace=False)  # type: np.ndarray

    under_sampled_features = np.vstack(
        (original_features[majority_sample_index, :], original_features[minority_index, :]))  # type: np.ndarray
    under_sampled_classes = np.hstack(
        (original_classes[majority_sample_index], original_classes[minority_index]))  # type: np.ndarray

    permutation_index = np.random.permutation(len(under_sampled_features))  # type:np.ndarray
    return under_sampled_features[permutation_index], under_sampled_classes[permutation_index]


def encode_training_data(sensor_data_train):
    # type: (np.ndarray) -> np.ndarray

    logging.info("Encoding categorical features...")
    encoder = OneHotEncoder(sparse=False)  # type: OneHotEncoder
    encoder.fit(sensor_data_train)

    with open(ENCODER_FILE, "wb") as encoder_file:
        pickle.dump(encoder, encoder_file)
        logging.info("Encoder saved at {}".format(ENCODER_FILE))

    sensor_data_train = encoder.transform(sensor_data_train)

    return sensor_data_train


def get_index_by_value(labels, label_value):
    # type: (np.ndarray, int) -> np.ndarray
    return np.where(labels == label_value)[0]


def train_type_analyser(sensor_data_train, person_type_train, batch_size, target_accuracy, encode_categorical_data,
                        units_per_layer, epochs=100, metric="binary_crossentropy", learning_rate=0.001):
    # type: (np.ndarray, np.ndarray, int, Optional[float], bool, List[int], int, str, float) -> SyntheticTypeAnalyser

    if encode_categorical_data:
        sensor_data_train = encode_training_data(sensor_data_train)

    _, num_features = sensor_data_train.shape
    logging.info("Training data shape: : {}. Learning rate {}".format(sensor_data_train.shape, learning_rate))

    type_analyser = SyntheticTypeAnalyser(num_features=num_features,
                                          metric=metric,
                                          learning_rate=learning_rate,
                                          units_per_layer=units_per_layer)  # type: SyntheticTypeAnalyser

    zero_responder_index = get_index_by_value(person_type_train,
                                              TYPE_TO_CLASS[SHARED_IDENTITY_TYPE])  # type: np.ndarray
    selfish_index = get_index_by_value(person_type_train, TYPE_TO_CLASS[PERSONAL_IDENTITY_TYPE])  # type: np.ndarray

    logging.info("Training data -> Zero-responders: %d" % len(zero_responder_index))
    logging.info("Training data -> Selfish: %d" % len(selfish_index))

    if len(zero_responder_index) != len(selfish_index):
        logging.info("Imbalanced dataset. Undersampling...")
        sensor_data_train, person_type_train = under_sample(first_index=zero_responder_index,
                                                            second_index=selfish_index,
                                                            original_features=sensor_data_train,
                                                            original_classes=person_type_train)
        zero_responder_index = get_index_by_value(person_type_train,
                                                  TYPE_TO_CLASS[SHARED_IDENTITY_TYPE])  # type: np.ndarray
        selfish_index = get_index_by_value(person_type_train, TYPE_TO_CLASS[PERSONAL_IDENTITY_TYPE])  # type: np.ndarray

        logging.info("Training data -> Zero-responders: %d" % len(zero_responder_index))
        logging.info("Training data -> Selfish: %d" % len(selfish_index))

    early_stopping_monitor = 'val_binary_crossentropy'
    if target_accuracy is not None:
        logging.info("Training for target accuracy %s" % str(target_accuracy))
        early_stopping_callback = EarlyStoppingByTarget(monitor=early_stopping_monitor, target=target_accuracy,
                                                        verbose=1)
    else:
        logging.info("Training for best accuracy")
        early_stopping_callback = EarlyStopping(monitor=early_stopping_monitor, patience=int(epochs * 0.1))

    # start_sanity_check(type_analyser, sensor_data_train, person_type_train, batch_size=batch_size)
    callbacks = [early_stopping_callback,
                 tf.keras.callbacks.TensorBoard(log_dir=get_log_directory()),
                 ModelCheckpoint(filepath=MODEL_FILE, monitor=early_stopping_monitor, save_best_only=True)]
    training_history = type_analyser.train(sensor_data_train,
                                           person_type_train,
                                           epochs,
                                           batch_size,
                                           callbacks)
    plot_training(training_history, metric)

    return type_analyser


def start_sanity_check(type_analyser, sensor_data_train, person_type_train, batch_size):
    # type: (SyntheticTypeAnalyser, np.ndarray, np.ndarray, int) -> None

    logging.info("SANITY CHECK: Single sample")
    zero_responder_index = get_index_by_value(person_type_train, TYPE_TO_CLASS[SHARED_IDENTITY_TYPE])  # type:np.ndarray

    single_sample_features = np.expand_dims(sensor_data_train[zero_responder_index[0]], axis=0)  # type:np.ndarray
    single_sample_type = np.expand_dims(person_type_train[zero_responder_index[0]], axis=0)  # type:np.ndarray

    training_history = type_analyser.do_sanity_check(single_sample_features,
                                                     single_sample_type,
                                                     batch_size=batch_size,
                                                     epochs=150)  # type: History
    plot_training(training_history, metric="acc", plot_validation=False)

    logging.info("SANITY CHECK: Many samples")
    sanity_check_samples = 10  # type:int
    sanity_check_index = np.random.choice(sensor_data_train.shape[0], size=sanity_check_samples)  # type:np.ndarray
    many_samples_features = sensor_data_train[sanity_check_index]  # type:np.ndarray
    many_samples_type = person_type_train[sanity_check_index]  # type:np.ndarray

    training_history = type_analyser.do_sanity_check(many_samples_features,
                                                     many_samples_type,
                                                     batch_size=batch_size,
                                                     epochs=1000)  # type: History
    plot_training(training_history, metric="acc", plot_validation=False)


def run_scenario(robot_controller, emergency_environment, num_scenarios):
    robot_payoffs = []

    for scenario in range(num_scenarios):

        current_sensor_data = emergency_environment.reset()
        done = False
        scenario_payoff = 0

        while not done:
            logging.info("Data Index: %d " % emergency_environment.data_index)
            robot_action = robot_controller.sensor_data_callback(current_sensor_data)
            logging.info("robot_action: %s" % robot_action)

            current_sensor_data, robot_payoff, done = emergency_environment.step(robot_action)
            scenario_payoff += robot_payoff

        robot_payoffs.append(scenario_payoff)

    logging.info("Scenarios: %.4f " % len(robot_payoffs))
    logging.info("Mean payoffs: %.4f " % np.mean(robot_payoffs))
    logging.info("Std payoffs: %.4f " % np.std(robot_payoffs))
    logging.info("Max payoffs: %.4f " % np.max(robot_payoffs))
    logging.info("Min payoffs: %.4f " % np.mean(robot_payoffs))

    return robot_payoffs


def main():
    zeroresponder_type_weight = 0.8  # According to: "Modelling social identification and helping in evacuation simulation"
    selfish_type_weight = 1 - zeroresponder_type_weight
    # target_accuracy = 0.65
    target_accuracy = None
    max_epochs = 10000  # type: int
    encode_categorical_data = True  # type: bool
    interactions_per_scenario = INTERACTIONS_PER_SCENARIO
    total_samples = 10000
    training_batch_size = 100
    num_scenarios = NUM_SCENARIOS
    train_analyser = True  # type:bool

    # sensor_data, person_type = get_netlogo_dataset()
    sensor_data, person_type = get_synthetic_dataset()
    sensor_data_train, sensor_data_test, person_type_train, person_type_test = train_test_split(sensor_data,
                                                                                                person_type,
                                                                                                test_size=0.33,
                                                                                                random_state=0)

    if train_analyser:
        type_analyser = train_type_analyser(sensor_data_train, person_type_train, training_batch_size, target_accuracy,
                                            encode_categorical_data, max_epochs)
    else:
        type_analyser = SyntheticTypeAnalyser(model_file=MODEL_FILE)

    robot_controller = AutonomicManagerController(type_analyser)

    # robot_controller = PessimisticRobotController()
    # robot_controller = OptimisticRobotController()

    # emergency_environment = EmergencyEvacuationEnvironment(sensor_data_test, person_type_test,
    #                                                        interactions_per_scenario)
    #
    # _ = run_scenario(robot_controller, emergency_environment, num_scenarios)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
