import gambit

ROBOT_PLAYER_INDEX = 0
PERSON_PLAYER_INDEX = 1


class InteractionGame(object):

    def __init__(self, title):
        self.game_tree = gambit.Game.new_tree()  # type: gambit.Game
        self.game_tree.title = title

        self.robot_player = self.game_tree.players.add("Robot")
        self.person_player = self.game_tree.players.add("Person")

        self.game_tree_root = self.game_tree.root
        self.type_nodes_per_label = {}
        self.robot_action_index_per_label = {}

    def configure_types(self, epistemic_types):
        chance_move = self.game_tree_root.append_move(self.game_tree.players.chance, len(epistemic_types))

        for index, type_information in enumerate(epistemic_types):
            description, prob_numerator, prob_denominator = type_information
            action = chance_move.actions[index]
            action.label = description
            action.prob = gambit.Rational(prob_numerator, prob_denominator)

            self.type_nodes_per_label[description] = self.game_tree_root.children[index]

    def set_first_contact_actions(self, robot_actions):
        first_type_node = self.game_tree_root.children[0]

        robot_firstcontact_move = first_type_node.append_move(self.robot_player, len(robot_actions))
        robot_firstcontact_move.label = "robot_first_move"
        robot_firstcontact_actions = robot_firstcontact_move.actions

        for index, robot_action in enumerate(robot_actions):
            robot_firstcontact_actions[index].label = robot_action
            self.robot_action_index_per_label[robot_action] = index

        for node_index in range(1, len(self.game_tree_root.children)):
            type_node = self.game_tree_root.children[node_index]
            type_node.append_move(robot_firstcontact_move)

    def configure_person_response(self, type, robot_action, person_responses):
        type_node = self.type_nodes_per_label[type]
        robot_action_index = self.robot_action_index_per_label[robot_action]

        person_response_node = type_node.children[robot_action_index]
        person_response_move = person_response_node.append_move(self.person_player, len(person_responses))
        person_response_move.label = type + "_on_" + robot_action + "_move"

        person_response_actions = person_response_move.actions
        for action_index, person_response in enumerate(person_responses):
            action_name, robot_payoff, person_payoff = person_response
            person_response_actions[action_index].label = action_name

            outcome = self.game_tree.outcomes.add(type + "_" + robot_action + "_" + action_name + "_outcome")
            outcome[ROBOT_PLAYER_INDEX] = robot_payoff
            outcome[PERSON_PLAYER_INDEX] = person_payoff
            action_node = person_response_node.children[action_index]
            action_node.outcome = outcome

    def write(self, show=True, filename="game_tree.efg"):
        game_as_efg = self.game_tree.write(format="efg")
        if show:
            print game_as_efg

        with open(filename, "w") as game_file:
            game_file.write(game_as_efg)


def main():
    interaction_game = InteractionGame("Two persons detected and one is a victim")
    interaction_game.configure_types([("ZERO_RESPONDER", 1, 2), ("SELFISH", 1, 2)])
    interaction_game.set_first_contact_actions(["follow_me", "wait_here"])

    interaction_game.configure_person_response("ZERO_RESPONDER", "follow_me",
                                               [("coming_together", 2, 2), ("coming_alone", -2, -2)])
    interaction_game.configure_person_response("ZERO_RESPONDER", "wait_here",
                                               [("wait_together", 1, 2), ("leave_alone", -1, -2)])
    interaction_game.configure_person_response("SELFISH", "follow_me",
                                               [("coming_together", 2, -1), ("coming_alone", -2, 2)])
    interaction_game.configure_person_response("SELFISH", "wait_here",
                                               [("wait_together", 1, -1), ("leave_alone", -1, 0)])
    interaction_game.write()


if __name__ == "__main__":
    main()
