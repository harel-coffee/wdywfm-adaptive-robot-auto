import logging

import solver
from samplegame import generate_game_model, WAIT_HERE_ROBOT_ACTION, FOLLOW_ME_ROBOT_ACTION


class AbstractRobotController(object):

    def sensor_data_callback(self, sensor_data):
        raise NotImplementedError("Subclasses must override sensor_data_callback")


class PessimisticRobotController(AbstractRobotController):

    def sensor_data_callback(self, sensor_data):
        return WAIT_HERE_ROBOT_ACTION


class OptimisticRobotController(AbstractRobotController):

    def sensor_data_callback(self, sensor_data):
        return FOLLOW_ME_ROBOT_ACTION


class AutonomicManagerController(AbstractRobotController):

    def __init__(self, type_analyser):
        self.type_analyser = type_analyser
        self.external_solver = solver.ExternalSubGamePerfectSolver()
        self.interaction_game = None

    def sensor_data_callback(self, sensor_data):
        type_probabilities = self.type_analyser.obtain_probabilities(sensor_data)
        group_identity_prob = type_probabilities.item()

        logging.info("group_identity_prob :  %.4f " % group_identity_prob)

        self.model_interaction(zero_responder_prob=group_identity_prob)
        equilibria = self.external_solver.solve(self.interaction_game.game_tree)

        if len(equilibria) == 0:
            logging.warning("No equilibria found! Aborting")
            return

        if len(equilibria) > 1:
            logging.warning("Multiple equilibria found! Aborting")
            return
        strategy_profile = equilibria[0]

        robot_strategy = self.interaction_game.get_robot_strategy(strategy_profile)
        robot_action = max(robot_strategy, key=robot_strategy.get)

        return robot_action

    def model_interaction(self, zero_responder_prob):
        zero_responder_ratio = zero_responder_prob.as_integer_ratio()
        selfish_ratio = (1 - zero_responder_prob).as_integer_ratio()

        self.interaction_game = generate_game_model(zero_responder_ratio, selfish_ratio, filename="controller_game.efg")
