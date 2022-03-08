import logging

import gambit

from game import InteractionGame

FOLLOW_ME_ROBOT_ACTION = "follow_me"
WAIT_HERE_ROBOT_ACTION = "wait_here"

GROUP_IDENTITY_TYPE = "GROUP_IDENTITY"
PERSONAL_IDENTITY_TYPE = "PERSONAL_IDENTITY"

# Identity impact = 0.3
# Unassisted success = 0.8
# Victim success = 0.2

FOLLOWME_SUCCESS_ROBOT_PAYOFF = 3  # Victim and Player with R + Extra with FR
FOLLOWME_FAIL_ROBOT_PAYOFF = gambit.Rational(11, 5)  # Extra with FR + Player with R + Victim unassisted
WAITHERE_SUCCESS_ROBOT_PAYOFF = gambit.Rational(14, 5)  # Victim and Player with FR + Extra unassisted
WAITHERE_FAIL_ROBOT_PAYOFF = gambit.Rational(13, 5)  # Victim with FR + Player unassisted + Extra unassisted


def generate_game_model(zero_responder_ratio, selfish_ratio, filename="game_file.efg"):
    logging.debug("zero_responder_ratio:  %s " % str(zero_responder_ratio))
    logging.debug("selfish_ratio:  %s " % str(selfish_ratio))

    interaction_game = InteractionGame("Two persons detected and one is a victim")
    interaction_game.configure_types(
        [(GROUP_IDENTITY_TYPE, zero_responder_ratio), (PERSONAL_IDENTITY_TYPE, selfish_ratio)])
    interaction_game.set_first_contact_actions([FOLLOW_ME_ROBOT_ACTION, WAIT_HERE_ROBOT_ACTION])

    interaction_game.configure_person_response(GROUP_IDENTITY_TYPE, FOLLOW_ME_ROBOT_ACTION,
                                               [(
                                                   "coming_together",
                                                   FOLLOWME_SUCCESS_ROBOT_PAYOFF,
                                                   gambit.Rational(13, 10)),  # 1 + Identity impact
                                                   ("coming_alone",
                                                    FOLLOWME_FAIL_ROBOT_PAYOFF,
                                                    gambit.Rational(7, 10))])  # 1 - Identity impact
    interaction_game.configure_person_response(GROUP_IDENTITY_TYPE, WAIT_HERE_ROBOT_ACTION,
                                               [("wait_together",
                                                 WAITHERE_SUCCESS_ROBOT_PAYOFF,
                                                 gambit.Rational(13, 10)),  # 1 + Identity impact
                                                ("leave_alone",
                                                 WAITHERE_FAIL_ROBOT_PAYOFF,
                                                 gambit.Rational(1, 2))])  # Unassisted success - Identity impact
    interaction_game.configure_person_response(PERSONAL_IDENTITY_TYPE, FOLLOW_ME_ROBOT_ACTION,
                                               [(
                                                   "coming_together",
                                                   FOLLOWME_SUCCESS_ROBOT_PAYOFF,
                                                   gambit.Rational(7, 10)),  # 1 - Identity impact
                                                   ("coming_alone",
                                                    FOLLOWME_FAIL_ROBOT_PAYOFF,
                                                    gambit.Rational(13, 10))])  # 1 + Identity impact
    interaction_game.configure_person_response(PERSONAL_IDENTITY_TYPE, WAIT_HERE_ROBOT_ACTION,
                                               [("wait_together",
                                                 WAITHERE_SUCCESS_ROBOT_PAYOFF,
                                                 gambit.Rational(7, 10)),  # 1 - Identity impact
                                                ("leave_alone",
                                                 WAITHERE_FAIL_ROBOT_PAYOFF,
                                                 gambit.Rational(11, 10))])  # Unassisted success + Identity impact

    if filename:
        interaction_game.write(filename)

    return interaction_game


def main():
    zero_responder_ratio = 0.5.as_integer_ratio()
    selfish_ratio = 0.5.as_integer_ratio()
    generate_game_model(zero_responder_ratio, selfish_ratio, filename="game_file.efg")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
