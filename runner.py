import logging

import solver
from game import InteractionGame


def main():
    interaction_game = InteractionGame("Two persons detected and one is a victim")
    interaction_game.configure_types([("ZERO_RESPONDER", 1, 10), ("SELFISH", 9, 10)])
    interaction_game.set_first_contact_actions(["follow_me", "wait_here"])

    interaction_game.configure_person_response("ZERO_RESPONDER", "follow_me",
                                               [("coming_together", 2, 2), ("coming_alone", -2, -2)])
    interaction_game.configure_person_response("ZERO_RESPONDER", "wait_here",
                                               [("wait_together", 1, 2), ("leave_alone", -1, -2)])
    interaction_game.configure_person_response("SELFISH", "follow_me",
                                               [("coming_together", 2, -1), ("coming_alone", -2, 2)])
    interaction_game.configure_person_response("SELFISH", "wait_here",
                                               [("wait_together", 1, -1), ("leave_alone", -1, 0)])
    _ = interaction_game.write()

    external_solver = solver.ExternalSubGamePerfectSolver()
    equilibria = external_solver.solve(interaction_game.game_tree)

    for strategy_profile in equilibria:
        print interaction_game.get_robot_strategy(strategy_profile)

        for person_type, _, _ in interaction_game.person_types:
            for robot_action in interaction_game.robot_actions:
                print "person_type", person_type, "robot_action", robot_action
                print interaction_game.get_person_strategy(strategy_profile, person_type, robot_action)

    outcome = interaction_game.get_outcome("SELFISH", "follow_me", "coming_alone")
    print outcome


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
