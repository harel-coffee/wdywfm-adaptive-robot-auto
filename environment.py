import logging

import solver
from game import InteractionGame, ROBOT_PLAYER_INDEX

CLASS_TO_TYPE = {
    0: "SELFISH",
    1: "ZERO_RESPONDER"
}


class EmergencyEnvironment(object):

    def __init__(self, sensor_data, person_type):
        self.data_index = 0
        _, features = sensor_data.shape

        self.num_features = features
        self.sensor_data = sensor_data
        self.person_type = person_type

        logging.info("Testing data: ", len(self.sensor_data))

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

        equilibria = self.external_solver.solve(self.interaction_game.game_tree)
        if len(equilibria) > 1:
            logging.warning("Multiple equilibria found! Aborting")
            return
        strategy_profile = equilibria[0]

        person_strategy = self.interaction_game.get_person_strategy(strategy_profile, current_person_type, robot_action)
        person_action = max(person_strategy, key=person_strategy.get)

        interaction_outcome = self.interaction_game.get_outcome(current_person_type, robot_action, person_action)
        robot_payoff = interaction_outcome[ROBOT_PLAYER_INDEX]

        next_observation = None
        done = False if self.data_index < len(self.sensor_data) else True

        if not done:
            self.data_index += 1
            next_observation = self.sensor_data[self.data_index]
            next_observation = next_observation.reshape(1, self.num_features)

            self.update_interaction_game()

        return next_observation, robot_payoff, done

    def update_interaction_game(self):
        current_type_class = self.person_type[self.data_index]
        current_person_type = CLASS_TO_TYPE[current_type_class]

        zero_responder_prob = 1. if current_person_type == "ZERO_RESPONDER" else 0.

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
