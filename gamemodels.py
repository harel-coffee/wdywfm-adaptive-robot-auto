import logging

import gambit
from typing import Tuple

from game import InteractionGame

ASK_FOR_HELP_ROBOT_ACTION = "ask-help"  # type:str
CALL_STAFF_ROBOT_ACTION = "call-staff"  # type:str
COOPERATE_PERSON_ACTION = "do-help"  # type:str
NOT_COOPERATE_PERSON_ACTION = "ignore"  # type:str

SHARED_IDENTITY_TYPE = "GROUP_IDENTITY"  # type:str
PERSONAL_IDENTITY_TYPE = "PERSONAL_IDENTITY"  # type:str


# Identity impact = 0.3
# Unassisted success = 0.8
# Victim success = 0.2


def generate_model_for_abm(zero_responder_ratio, selfish_ratio, filename="game_file.efg"):
    # type: (Tuple[int, int], Tuple[int, int], str) -> InteractionGame

    logging.debug("zero_responder_ratio:  %s " % str(zero_responder_ratio))
    logging.debug("selfish_ratio:  %s " % str(selfish_ratio))

    interaction_game = InteractionGame("Two persons detected and one is a victim")  # type: InteractionGame
    interaction_game.configure_types(
        [(SHARED_IDENTITY_TYPE, zero_responder_ratio), (PERSONAL_IDENTITY_TYPE, selfish_ratio)])
    interaction_game.set_first_contact_actions([ASK_FOR_HELP_ROBOT_ACTION, CALL_STAFF_ROBOT_ACTION])

    person_payoff_identity_compliant = gambit.Rational(1, 1)  # type: gambit.Rational
    person_payoff_against_identity = gambit.Rational(-1, 1)  # type: gambit.Rational
    person_payoff_not_contacted = gambit.Rational(0, 1)  # type: gambit.Rational

    robot_payoff_with_support = gambit.Rational(3, 1)  # type: gambit.Rational
    robot_payoff_request_ignored = gambit.Rational(-1, 1)  # type: gambit.Rational
    robot_payoff_call_staff = gambit.Rational(0, 1)  # type: gambit.Rational

    interaction_game.configure_person_response(SHARED_IDENTITY_TYPE, ASK_FOR_HELP_ROBOT_ACTION,
                                               [(
                                                   COOPERATE_PERSON_ACTION,
                                                   robot_payoff_with_support,
                                                   person_payoff_identity_compliant
                                               ), (
                                                   NOT_COOPERATE_PERSON_ACTION,
                                                   robot_payoff_request_ignored,
                                                   person_payoff_against_identity
                                               )])
    interaction_game.configure_final_outcome(SHARED_IDENTITY_TYPE, CALL_STAFF_ROBOT_ACTION,
                                             (
                                                 robot_payoff_call_staff,
                                                 person_payoff_not_contacted
                                             ))
    interaction_game.configure_person_response(PERSONAL_IDENTITY_TYPE, ASK_FOR_HELP_ROBOT_ACTION,
                                               [(
                                                   COOPERATE_PERSON_ACTION,
                                                   robot_payoff_with_support,
                                                   person_payoff_against_identity
                                               ), (
                                                   NOT_COOPERATE_PERSON_ACTION,
                                                   robot_payoff_request_ignored,
                                                   person_payoff_identity_compliant
                                               )])
    interaction_game.configure_final_outcome(PERSONAL_IDENTITY_TYPE, CALL_STAFF_ROBOT_ACTION,
                                             (
                                                 robot_payoff_call_staff,
                                                 person_payoff_not_contacted
                                             ))

    if filename:
        interaction_game.write(filename)

    return interaction_game


def generate_game_model(zero_responder_ratio, selfish_ratio, filename="game_file.efg"):
    # type: (Tuple[int, int], Tuple[int, int], str) -> InteractionGame

    askhelp_success_robot_payoff = 3  # Victim and Player with R + Extra with FR
    askhelp_fail_robot_payoff = gambit.Rational(11, 5)  # Extra with FR + Player with R + Victim unassisted
    callstaff_success_robot_payoff = gambit.Rational(14, 5)  # Victim and Player with FR + Extra unassisted
    callstaff_fail_robot_payoff = gambit.Rational(13, 5)  # Victim with FR + Player unassisted + Extra unassisted

    logging.debug("zero_responder_ratio:  %s " % str(zero_responder_ratio))
    logging.debug("selfish_ratio:  %s " % str(selfish_ratio))

    interaction_game = InteractionGame("Two persons detected and one is a victim")  # type: InteractionGame
    interaction_game.configure_types(
        [(SHARED_IDENTITY_TYPE, zero_responder_ratio), (PERSONAL_IDENTITY_TYPE, selfish_ratio)])
    interaction_game.set_first_contact_actions([ASK_FOR_HELP_ROBOT_ACTION, CALL_STAFF_ROBOT_ACTION])

    interaction_game.configure_person_response(SHARED_IDENTITY_TYPE, ASK_FOR_HELP_ROBOT_ACTION,
                                               [(
                                                   COOPERATE_PERSON_ACTION,
                                                   askhelp_success_robot_payoff,
                                                   gambit.Rational(13, 10)  # 1 + Identity impact
                                               ), (
                                                   NOT_COOPERATE_PERSON_ACTION,
                                                   askhelp_fail_robot_payoff,
                                                   gambit.Rational(7, 10)  # 1 - Identity impact
                                               )])
    interaction_game.configure_person_response(SHARED_IDENTITY_TYPE, CALL_STAFF_ROBOT_ACTION,
                                               [(
                                                   "wait_together",
                                                   callstaff_success_robot_payoff,
                                                   gambit.Rational(13, 10)  # 1 + Identity impact
                                               ), (
                                                   "leave_alone",
                                                   callstaff_fail_robot_payoff,
                                                   gambit.Rational(1, 2)  # Unassisted success - Identity impact
                                               )])
    interaction_game.configure_person_response(PERSONAL_IDENTITY_TYPE, ASK_FOR_HELP_ROBOT_ACTION,
                                               [(
                                                   COOPERATE_PERSON_ACTION,
                                                   askhelp_success_robot_payoff,
                                                   gambit.Rational(7, 10)  # 1 - Identity impact
                                               ), (
                                                   NOT_COOPERATE_PERSON_ACTION,
                                                   askhelp_fail_robot_payoff,
                                                   gambit.Rational(13, 10)  # 1 + Identity impact
                                               )])
    interaction_game.configure_person_response(PERSONAL_IDENTITY_TYPE, CALL_STAFF_ROBOT_ACTION,
                                               [(
                                                   "wait_together",
                                                   callstaff_success_robot_payoff,
                                                   gambit.Rational(7, 10)  # 1 - Identity impact
                                               ), (
                                                   "leave_alone",
                                                   callstaff_fail_robot_payoff,
                                                   gambit.Rational(11, 10)  # Unassisted success + Identity impact
                                               )])

    if filename:
        interaction_game.write(filename)

    return interaction_game


def main():
    zero_responder_ratio = 0.3.as_integer_ratio()
    selfish_ratio = 0.7.as_integer_ratio()
    generate_model_for_abm(zero_responder_ratio, selfish_ratio, filename="game_file.efg")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
