import logging

import solver
from game import ROBOT_PLAYER_INDEX, PERSON_PLAYER_INDEX
from samplegame import generate_game_model, SELFISH_TYPE, ZERO_RESPONDER_TYPE

CLASS_TO_TYPE = {
    0: SELFISH_TYPE,
    1: ZERO_RESPONDER_TYPE
}


class EmergencyEnvironment(object):

    def __init__(self, sensor_data, person_type):
        self.data_index = 0
        _, features = sensor_data.shape

        self.num_features = features
        self.sensor_data = sensor_data
        self.person_type = person_type

        logging.info("Testing data:  %.4f " % len(self.sensor_data))

        self.interaction_game = None
        self.external_solver = solver.ExternalSubGamePerfectSolver()

    def reset(self):
        self.data_index = 0
        next_observation = self.sensor_data[self.data_index]
        next_observation = next_observation.reshape(1, self.num_features)

        self.update_interaction_game()
        return next_observation

    def step(self, robot_action):

        current_type_class = self.person_type[self.data_index]
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
        robot_payoff = interaction_outcome[ROBOT_PLAYER_INDEX]
        person_payoff = interaction_outcome[PERSON_PLAYER_INDEX]

        logging.info("robot_payoff: %.4f" % robot_payoff)
        logging.info("person_payoff: %.4f" % person_payoff)

        next_observation = None
        done = False if self.data_index < (len(self.sensor_data) - 1) else True

        if not done:
            self.data_index += 1
            next_observation = self.sensor_data[self.data_index]
            next_observation = next_observation.reshape(1, self.num_features)

            self.update_interaction_game()

        return next_observation, robot_payoff, done

    def update_interaction_game(self):
        current_type_class = self.person_type[self.data_index]
        current_person_type = CLASS_TO_TYPE[current_type_class]

        zero_responder_prob = 1. if current_person_type == ZERO_RESPONDER_TYPE else 0.

        zero_responder_ratio = zero_responder_prob.as_integer_ratio()
        selfish_ratio = (1 - zero_responder_prob).as_integer_ratio()

        self.interaction_game = generate_game_model(zero_responder_ratio, selfish_ratio,
                                                    filename="environment_game.efg")
