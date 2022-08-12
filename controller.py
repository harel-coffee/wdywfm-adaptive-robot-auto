import logging

import numpy as np
from gambit.nash import NashSolution
from typing import Optional, Tuple, List, Dict

import analyser
import solver
from game import InteractionGame
from gamemodels import CALL_STAFF_ROBOT_ACTION, ASK_FOR_HELP_ROBOT_ACTION, generate_model_for_abm


class AbstractRobotController(object):

    def sensor_data_callback(self, sensor_data):
        raise NotImplementedError("Subclasses must override sensor_data_callback")


class ProSelfRobotController(AbstractRobotController):

    def sensor_data_callback(self, sensor_data):
        return CALL_STAFF_ROBOT_ACTION


class ProSocialRobotController(AbstractRobotController):

    def sensor_data_callback(self, sensor_data):
        return ASK_FOR_HELP_ROBOT_ACTION


class AutonomicManagerController(AbstractRobotController):

    def __init__(self, type_analyser):
        self.type_analyser = type_analyser  # type: analyser.SyntheticTypeAnalyser
        self.external_solver = solver.ExternalSubGamePerfectSolver()  # type: solver.ExternalSubGamePerfectSolver
        self.interaction_game = None  # type: Optional[InteractionGame]

    def get_shared_identity_probability(self, sensor_data):
        # type: (np.ndarray) -> float

        type_probabilities = self.type_analyser.obtain_probabilities(sensor_data)  # type: np.ndarray
        shared_identity_prob = type_probabilities.item()  # type: float

        return shared_identity_prob

    def sensor_data_callback(self, sensor_data):
        # type: (np.ndarray) -> Optional[str]

        group_identity_prob = self.get_shared_identity_probability(sensor_data)  # type: float
        logging.info("group_identity_prob :  %.4f " % group_identity_prob)

        self.model_interaction(zero_responder_prob=group_identity_prob)
        equilibria = self.external_solver.solve(self.interaction_game.game_tree)  # type: List[NashSolution]

        if len(equilibria) == 0:
            logging.warning("No equilibria found! Aborting")
            return

        if len(equilibria) > 1:
            logging.warning("Multiple equilibria found! Aborting")
            return
        strategy_profile = equilibria[0]  # type: NashSolution

        robot_strategy = self.interaction_game.get_robot_strategy(strategy_profile)  # type: Dict[str, float]
        robot_action = max(robot_strategy, key=robot_strategy.get)  # type: str

        return robot_action

    def model_interaction(self, zero_responder_prob):
        # type: (float) -> None

        zero_responder_ratio = zero_responder_prob.as_integer_ratio()  # type: Tuple [int, int]
        selfish_ratio = (1 - zero_responder_prob).as_integer_ratio()  # type: Tuple [int, int]

        self.interaction_game = generate_model_for_abm(zero_responder_ratio, selfish_ratio,
                                                       filename="controller_game.efg")


def main():
    manager = AutonomicManagerController(analyser.SyntheticTypeAnalyser(model_file="trained_model.h5"))
    sample_sensor_reading = np.zeros(shape=(1, 31))  # type: np.ndarray
    robot_action = manager.sensor_data_callback(sample_sensor_reading)
    print(robot_action)


if __name__ == "__main__":
    main()
