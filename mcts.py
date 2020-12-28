

class MCTS:
    def __init__(self, game_rules, nnet, args):
        self.game_rules = game_rules
        self.nnet = nnet
        self.args = args

        self.Qsa = []
        self.Nsa = []
        self.Ps = []

    def tree_search(self, board, t):
        for _ in range(self.args.monte_carlo_simulations):
            self.simulate(board)

        s = self.game_rules.tostring(board)
        return []

    def simulate(self, board):
        pass