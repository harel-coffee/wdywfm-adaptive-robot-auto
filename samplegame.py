import logging

from game import InteractionGame


def generate_game_model(zero_responder_ratio, selfish_ratio, filename="game_file.efg"):
    logging.debug("zero_responder_ratio:  %s " % str(zero_responder_ratio))
    logging.debug("selfish_ratio:  %s " % str(selfish_ratio))

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

    if filename:
        interaction_game.write(filename)

    return interaction_game
