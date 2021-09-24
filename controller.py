import logging

import solver
from samplegame import generate_game_model


class RobotController(object):

    def __init__(self, type_analyser):
        self.type_analyser = type_analyser
        self.external_solver = solver.ExternalSubGamePerfectSolver()
        self.interaction_game = None

    def sensor_data_callback(self, sensor_data):
        type_probabilities = self.type_analyser.obtain_probabilities(sensor_data)
        zero_responder_prob = type_probabilities.item()

        logging.info("zero_responder_prob :  %.4f " % zero_responder_prob)

        self.model_interaction(zero_responder_prob=zero_responder_prob)
        equilibria = self.external_solver.solve(self.interaction_game.game_tree)

        if len(equilibria) > 1:
            logging.warning("Multiple equilibria found! Aborting")
            return
        strategy_profile = equilibria[0]

        robot_strategy = self.interaction_game.get_robot_strategy(strategy_profile)
        robot_action = max(robot_strategy, key=robot_strategy.get)

        logging.info("robot_action: %s" % robot_action)

        return robot_action

    def model_interaction(self, zero_responder_prob):
        zero_responder_ratio = zero_responder_prob.as_integer_ratio()
        selfish_ratio = (1 - zero_responder_prob).as_integer_ratio()

        self.interaction_game = generate_game_model(zero_responder_ratio, selfish_ratio, filename="controller_game.efg")
