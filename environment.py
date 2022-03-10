import logging

import numpy as np

import solver
from game import ROBOT_PLAYER_INDEX, PERSON_PLAYER_INDEX
from samplegame import generate_game_model, PERSONAL_IDENTITY_TYPE, GROUP_IDENTITY_TYPE

PERSONAL_IDENTITY_CLASS = 0
GROUP_IDENTITY_CLASS = 1

CLASS_TO_TYPE = {
    PERSONAL_IDENTITY_CLASS: PERSONAL_IDENTITY_TYPE,
    GROUP_IDENTITY_CLASS: GROUP_IDENTITY_TYPE
}


class EmergencyEvacuationEnvironment(object):

    def __init__(self, sensor_data, person_type, interactions_per_scenario=33):

        self.data_index = 0
        self.interactions_per_scenario = interactions_per_scenario
        self.total_sensor_data = sensor_data
        self.total_person_type = person_type
        logging.info("Testing data:  {} ".format(self.total_sensor_data.shape))

        _, features = sensor_data.shape

        self.num_features = features
        self.sample_sensor_data = None
        self.sample_person_type = None

        self.interaction_game = None
        self.external_solver = solver.ExternalSubGamePerfectSolver()

    def configure_scenario(self):
        records, _ = self.total_sensor_data.shape
        selection_index = range(0, records)

        index_current_sample = np.random.choice(selection_index, size=self.interactions_per_scenario, replace=False)
        self.sample_sensor_data = self.total_sensor_data[index_current_sample, :]
        self.sample_person_type = self.total_person_type[index_current_sample]

        logging.debug("Samples for scenario : {}".format(self.sample_sensor_data.shape))

    def reset(self):
        logging.info("Starting a new scenario")
        self.data_index = 0
        self.configure_scenario()

        next_observation = self.sample_sensor_data[self.data_index]
        next_observation = next_observation.reshape(1, self.num_features)

        self.update_interaction_game()
        return next_observation

    def step(self, robot_action):

        current_type_class = self.sample_person_type[self.data_index]
        current_person_type = CLASS_TO_TYPE[current_type_class]

        logging.info("current_person_type: %s" % current_person_type)
        equilibria = self.external_solver.solve(self.interaction_game.game_tree)
        if len(equilibria) > 1:
            logging.warning("Multiple equilibria found! Aborting")
            return
        strategy_profile = equilibria[0]

        person_strategy = self.interaction_game.get_person_strategy(strategy_profile, current_person_type, robot_action)
        person_action = max(person_strategy, key=person_strategy.get)
        logging.info("person_action: %s " % person_action)

        interaction_outcome = self.interaction_game.get_outcome(current_person_type, robot_action, person_action)
        robot_payoff = float(interaction_outcome[ROBOT_PLAYER_INDEX])
        person_payoff = float(interaction_outcome[PERSON_PLAYER_INDEX])

        logging.info("robot_payoff: %.4f" % robot_payoff)
        logging.info("person_payoff: %.4f" % person_payoff)

        next_observation = None
        done = False if self.data_index < (len(self.sample_sensor_data) - 1) else True

        if not done:
            self.data_index += 1
            next_observation = self.sample_sensor_data[self.data_index]
            next_observation = next_observation.reshape(1, self.num_features)

            self.update_interaction_game()
        else:
            logging.info("Scenario finished")

        return next_observation, robot_payoff, done

    def update_interaction_game(self):
        current_type_class = self.sample_person_type[self.data_index]
        current_person_type = CLASS_TO_TYPE[current_type_class]

        zero_responder_prob = 1. if current_person_type == GROUP_IDENTITY_TYPE else 0.

        zero_responder_ratio = zero_responder_prob.as_integer_ratio()
        selfish_ratio = (1 - zero_responder_prob).as_integer_ratio()

        self.interaction_game = generate_game_model(zero_responder_ratio, selfish_ratio,
                                                    filename="environment_game.efg")
