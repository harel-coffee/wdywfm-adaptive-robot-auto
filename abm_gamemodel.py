import logging

import gambit
from typing import Tuple, Optional

from game import InteractionGame
from gamemodel import SHARED_IDENTITY_TYPE, PERSONAL_IDENTITY_TYPE, ASK_FOR_HELP_ROBOT_ACTION, CALL_STAFF_ROBOT_ACTION, \
    COOPERATE_PERSON_ACTION, NOT_COOPERATE_PERSON_ACTION


def generate_game_model(zero_responder_ratio, selfish_ratio, filename):
    # type: (Tuple[int, int], Tuple[int, int], Optional[str]) -> InteractionGame

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
