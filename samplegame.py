import logging

from game import InteractionGame

FOLLOW_ME_ROBOT_ACTION = "follow_me"
WAIT_HERE_ROBOT_ACTION = "wait_here"

ZERO_RESPONDER_TYPE = "ZERO_RESPONDER"
SELFISH_TYPE = "SELFISH"


def generate_game_model(zero_responder_ratio, selfish_ratio, filename="game_file.efg"):
    logging.debug("zero_responder_ratio:  %s " % str(zero_responder_ratio))
    logging.debug("selfish_ratio:  %s " % str(selfish_ratio))

    interaction_game = InteractionGame("Two persons detected and one is a victim")
    interaction_game.configure_types([(ZERO_RESPONDER_TYPE, zero_responder_ratio), (SELFISH_TYPE, selfish_ratio)])
    interaction_game.set_first_contact_actions([FOLLOW_ME_ROBOT_ACTION, WAIT_HERE_ROBOT_ACTION])

    interaction_game.configure_person_response(ZERO_RESPONDER_TYPE, FOLLOW_ME_ROBOT_ACTION,
                                               [("coming_together", 2, 2), ("coming_alone", -2, -2)])
    interaction_game.configure_person_response(ZERO_RESPONDER_TYPE, WAIT_HERE_ROBOT_ACTION,
                                               [("wait_together", 1, 2), ("leave_alone", -1, -2)])
    interaction_game.configure_person_response(SELFISH_TYPE, FOLLOW_ME_ROBOT_ACTION,
                                               [("coming_together", 2, -1), ("coming_alone", -2, 2)])
    interaction_game.configure_person_response(SELFISH_TYPE, WAIT_HERE_ROBOT_ACTION,
                                               [("wait_together", 1, -1), ("leave_alone", -1, 0)])

    if filename:
        interaction_game.write(filename)

    return interaction_game
