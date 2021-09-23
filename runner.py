import logging

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import solver
from analyser import TypeAnalyser
from game import InteractionGame

CLASS_TO_TYPE = {
    0: "SELFISH",
    1: "ZERO_RESPONDER"
}


def get_interaction_game(zero_responder_prob, write=True):
    zero_responder_ratio = zero_responder_prob.as_integer_ratio()
    selfish_ratio = (1 - zero_responder_prob).as_integer_ratio()

    interaction_game = InteractionGame("Two persons detected and one is a victim")
    interaction_game.configure_types([("ZERO_RESPONDER", zero_responder_ratio), ("SELFISH", selfish_ratio)])
    interaction_game.set_first_contact_actions(["follow_me", "wait_here"])

    interaction_game.configure_person_response("ZERO_RESPONDER", "follow_me",
                                               [("coming_together", 2, 2), ("coming_alone", -2, -2)])
    interaction_game.configure_person_response("ZERO_RESPONDER", "wait_here",
                                               [("wait_together", 1, 2), ("leave_alone", -1, -2)])
    interaction_game.configure_person_response("SELFISH", "follow_me",
                                               [("coming_together", 2, -1), ("coming_alone", -2, 2)])
    interaction_game.configure_person_response("SELFISH", "wait_here",
                                               [("wait_together", 1, -1), ("leave_alone", -1, 0)])

    if write:
        interaction_game.write()

    return interaction_game


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
    logging.info("Training data: ", len(sensor_data_train))
    logging.info("Testing data: ", len(sensor_data_test))
    _, num_features = sensor_data.shape
    type_analyser = TypeAnalyser(num_features)
    training_history = type_analyser.train(sensor_data_train, person_type_train)
    plot_training(training_history, "acc")

    data_index = 0
    current_type_class = person_type_test[data_index]
    current_person_type = CLASS_TO_TYPE[current_type_class]
    current_sensor_data = sensor_data_test[data_index]
    current_sensor_data = current_sensor_data.reshape(1, num_features)
    type_probabilities = type_analyser.obtain_probabilities(current_sensor_data)
    zero_responder_prob = type_probabilities.item()

    interaction_game = get_interaction_game(zero_responder_prob=zero_responder_prob)

    external_solver = solver.ExternalSubGamePerfectSolver()
    equilibria = external_solver.solve(interaction_game.game_tree)
    if len(equilibria) > 1:
        logging.warning("Multiple equilibria found! Aborting")
        return
    strategy_profile = equilibria[0]

    robot_strategy = interaction_game.get_robot_strategy(strategy_profile)
    robot_action = max(robot_strategy, key=robot_strategy.get)
    person_strategy = interaction_game.get_person_strategy(strategy_profile, current_person_type, robot_action)
    person_action = max(person_strategy, key=person_strategy.get)
    interaction_outcome = interaction_game.get_outcome(current_person_type, robot_action, person_action)

    print interaction_outcome


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
