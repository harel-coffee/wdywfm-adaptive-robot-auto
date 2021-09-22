from gambit.nash import ExternalSolver


class ExternalSubGamePerfectSolver(ExternalSolver):

    def solve(self, game):
        if not game.is_perfect_recall:
            raise RuntimeError("Computing equilibria of games with imperfect recall is not supported.")
        if not game.is_tree:
            raise RuntimeError("This solver has no effect on strategic games.")

        command_line = "gambit-enumpure -P"

        return self._parse_output(self.launch(command_line, game), game, rational=True, extensive=game.is_tree)
