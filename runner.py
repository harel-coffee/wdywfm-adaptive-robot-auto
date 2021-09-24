import logging

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from analyser import TypeAnalyser
from controller import RobotController
from environment import EmergencyEnvironment


def get_dataset(selfish_type_weight, zeroresponder_type_weight):
    features, target = make_classification(n_samples=10000,
                                           n_features=100,
                                           n_informative=3,
                                           n_redundant=0,
                                           n_classes=2,
                                           weights=[selfish_type_weight, zeroresponder_type_weight],
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

    selfish_type_weight = 0.5
    zeroresponder_type_weight = 0.5

    sensor_data, person_type = get_dataset(selfish_type_weight, zeroresponder_type_weight)
    sensor_data_train, sensor_data_test, person_type_train, person_type_test = train_test_split(sensor_data,
                                                                                                person_type,
                                                                                                test_size=0.33,
                                                                                                random_state=0)
    logging.info("Training data: : %.4f" % len(sensor_data_train))
    _, num_features = sensor_data.shape
    type_analyser = TypeAnalyser(num_features)
    training_history = type_analyser.train(sensor_data_train, person_type_train)
    plot_training(training_history, "acc")

    robot_controller = RobotController(type_analyser)
    emergency_environment = EmergencyEnvironment(sensor_data_test, person_type_test)

    current_sensor_data = emergency_environment.reset()
    done = False

    robot_payoffs = []
    while not done:
        logging.info("Data Index: %d " % emergency_environment.data_index)
        robot_action = robot_controller.sensor_data_callback(current_sensor_data)
        next_observation, robot_payoff, done = emergency_environment.step(robot_action)
        robot_payoffs.append(robot_payoff)

    logging.info("Interactions: %.4f " % len(robot_payoffs))
    logging.info("Total payoffs: %.4f " % sum(robot_payoffs))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
