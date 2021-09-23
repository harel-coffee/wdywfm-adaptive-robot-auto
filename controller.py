import logging

import solver
from game import InteractionGame


class RobotController(object):

    def __init__(self, type_analyser):
        self.type_analyser = type_analyser
        self.external_solver = solver.ExternalSubGamePerfectSolver()
        self.interaction_game = None

    def sensor_data_callback(self, sensor_data):
        type_probabilities = self.type_analyser.obtain_probabilities(sensor_data)
        zero_responder_prob = type_probabilities.item()

        self.model_interaction(zero_responder_prob=zero_responder_prob)
        equilibria = self.external_solver.solve(self.interaction_game.game_tree)

        if len(equilibria) > 1:
            logging.warning("Multiple equilibria found! Aborting")
            return
        strategy_profile = equilibria[0]

        robot_strategy = self.interaction_game.get_robot_strategy(strategy_profile)
        robot_action = max(robot_strategy, key=robot_strategy.get)

        return robot_action

    def model_interaction(self, zero_responder_prob, write=True):
        zero_responder_ratio = zero_responder_prob.as_integer_ratio()
        selfish_ratio = (1 - zero_responder_prob).as_integer_ratio()

        self.interaction_game = InteractionGame("Two persons detected and one is a victim")
        self.interaction_game.configure_types([("ZERO_RESPONDER", zero_responder_ratio), ("SELFISH", selfish_ratio)])
        self.interaction_game.set_first_contact_actions(["follow_me", "wait_here"])

        self.interaction_game.configure_person_response("ZERO_RESPONDER", "follow_me",
                                                        [("coming_together", 2, 2), ("coming_alone", -2, -2)])
        self.interaction_game.configure_person_response("ZERO_RESPONDER", "wait_here",
                                                        [("wait_together", 1, 2), ("leave_alone", -1, -2)])
        self.interaction_game.configure_person_response("SELFISH", "follow_me",
                                                        [("coming_together", 2, -1), ("coming_alone", -2, 2)])
        self.interaction_game.configure_person_response("SELFISH", "wait_here",
                                                        [("wait_together", 1, -1), ("leave_alone", -1, 0)])

        if write:
            self.interaction_game.write()
