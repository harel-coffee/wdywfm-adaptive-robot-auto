import logging

import gambit
from gambit import Rational
from gambit.nash import NashSolution
from typing import Dict, List, Tuple, Any, Optional

ROBOT_PLAYER_INDEX = 0
PERSON_PLAYER_INDEX = 1


class InteractionGame(object):

    def __init__(self, title):
        self.game_tree = gambit.Game.new_tree()  # type: gambit.Game
        self.game_tree.title = title

        self.robot_player = self.game_tree.players.add("Robot")
        self.robot_actions = []
        self.robot_firstcontact_infoset = None

        self.person_player = self.game_tree.players.add("Person")
        self.person_types = []
        self.person_response_infosets = {}  # type: Dict[Tuple[str, str], Any]
        self.person_responses = {}  # type: Dict[Tuple[str, str], Any]

        self.game_tree_root = self.game_tree.root
        self.type_nodes_per_label = {}  # type: Dict[str, Any]
        self.robot_action_index_per_label = {}  # type: Dict[str, int]

    def configure_types(self, epistemic_types):
        # type: (List[Tuple[str, Tuple]]) -> None

        chance_move = self.game_tree_root.append_move(self.game_tree.players.chance, len(epistemic_types))
        self.person_types = epistemic_types

        for index, type_information in enumerate(self.person_types):
            description, probability = type_information
            action = chance_move.actions[index]
            action.label = description
            action.prob = gambit.Rational(probability[0], probability[1])

            self.type_nodes_per_label[description] = self.game_tree_root.children[index]

    def set_first_contact_actions(self, robot_actions):
        # type: (List[str]) -> None

        self.robot_actions = robot_actions
        first_type_node = self.game_tree_root.children[0]

        self.robot_firstcontact_infoset = first_type_node.append_move(self.robot_player, len(self.robot_actions))
        self.robot_firstcontact_infoset.label = "robot_first_move"
        robot_firstcontact_actions = self.robot_firstcontact_infoset.actions

        for index, robot_action in enumerate(self.robot_actions):
            robot_firstcontact_actions[index].label = robot_action
            self.robot_action_index_per_label[robot_action] = index

        for node_index in range(1, len(self.game_tree_root.children)):
            type_node = self.game_tree_root.children[node_index]
            type_node.append_move(self.robot_firstcontact_infoset)

    def configure_final_outcome(self, person_type, robot_action, payoffs):
        # type: (str, str, Tuple[Rational, Rational]) -> None

        robot_payoff, person_payoff = payoffs

        type_node = self.type_nodes_per_label[person_type]
        robot_action_index = self.robot_action_index_per_label[robot_action]  # type: int
        robot_response_node = type_node.children[robot_action_index]

        outcome = self.game_tree.outcomes.add(self.get_outcome_description(person_type, robot_action))
        outcome[ROBOT_PLAYER_INDEX] = robot_payoff
        outcome[PERSON_PLAYER_INDEX] = person_payoff

        robot_response_node.outcome = outcome

    def configure_person_response(self, person_type, robot_action, responses):
        # type: (str, str, List[Tuple[Optional[str], Rational, Rational]]) -> None

        type_node = self.type_nodes_per_label[person_type]
        robot_action_index = self.robot_action_index_per_label[robot_action]  # type: int
        person_response_node = type_node.children[robot_action_index]

        person_response_infoset = person_response_node.append_move(self.person_player, len(responses))
        self.person_response_infosets[(person_type, robot_action)] = person_response_infoset
        self.person_responses[(person_type, robot_action)] = responses

        person_response_infoset.label = person_type + "_on_" + robot_action + "_move"

        person_response_actions = person_response_infoset.actions
        for action_index, person_response in enumerate(responses):
            action_name, robot_payoff, person_payoff = person_response
            person_response_actions[action_index].label = action_name

            outcome = self.game_tree.outcomes.add(self.get_outcome_description(person_type, robot_action, action_name))
            outcome[ROBOT_PLAYER_INDEX] = robot_payoff
            outcome[PERSON_PLAYER_INDEX] = person_payoff
            action_node = person_response_node.children[action_index]
            action_node.outcome = outcome

    @staticmethod
    def get_outcome_description(person_type, robot_action, person_response=None):
        # type: (str, str, Optional[str]) -> str

        if not person_response:
            person_response = "no_response"
        return person_type + "_" + robot_action + "_" + person_response + "_outcome"

    def write(self, filename="game_tree.efg"):
        # type: (str) -> str
        game_as_efg = self.game_tree.write(format="efg")
        logging.debug(game_as_efg)

        with open(filename, "w") as game_file:
            game_file.write(game_as_efg)

        logging.debug("Game written to %s" % filename)
        return filename

    def get_robot_strategy(self, strategy_profile):
        # type: (NashSolution) -> Dict[str, float]

        action_probabilities = strategy_profile[self.robot_firstcontact_infoset]
        strategy = {}
        for action_index, action_probability in enumerate(action_probabilities):
            strategy[self.robot_actions[action_index]] = action_probability

        return strategy

    def get_person_strategy(self, strategy_profile, person_type, robot_action):
        action_probabilities = strategy_profile[self.person_response_infosets[(person_type, robot_action)]]
        person_actions = [action for action, _, _ in self.person_responses[(person_type, robot_action)]]

        strategy = {}
        for action_index, action_probability in enumerate(action_probabilities):
            strategy[person_actions[action_index]] = action_probability

        return strategy

    def get_outcome(self, person_type, robot_action, person_response):
        outcomes = [outcome for outcome in self.game_tree.outcomes if
                    outcome.label == self.get_outcome_description(person_type, robot_action, person_response)]
        if len(outcomes) != 1:
            logging.error("Multiple outcomes found")
            return None

        return outcomes[0][ROBOT_PLAYER_INDEX], outcomes[0][PERSON_PLAYER_INDEX]
