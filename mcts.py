import numpy as np
from collections import defaultdict

class MCTS:
    def __init__(self, game_rules, nnet, args):
        self.game_rules = game_rules
        self.nnet = nnet
        self.args = args

        self.Qsa = defaultdict(float)
        self.Nsa = defaultdict(float)
        self.Ns = defaultdict(lambda: 1e-8)
        self.Ps = defaultdict(float)

    def tree_search(self, board, temperature):
        # run monte carlo tree search simulations
        for _ in range(self.args.monte_carlo_simulations):
            self.simulate(board)

        s = self.game_rules.tostring(board)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game_rules.get_action_space())]

        if temperature == 0:
            best_actions = np.array(np.where(counts == np.max(counts))).flatten()
            best_action = np.random.choice(best_actions)
            probs = [0] * len(counts)
            probs[best_action] = 1
            return probs

        counts = [x ** (1. / temperature) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def simulate(self, board):
        # if the game state is terminal
        if self.game_rules.terminal(board):
            value = self.game_rules.result(board, 1)
            return -value

        s = self.game_rules.tostring(board)
        valid_actions = self.game_rules.get_valid_actions(board)

        # if we haven't evaluated this position before (we have reached a leaf node)
        if s not in self.Ps:
            pi, value = self.nnet.evaluate(board)
            pi = pi * valid_actions
            pi = pi / np.sum(pi)
            self.Ps[s] = pi
            return -value

        # find the action that maximizes the upper confidence bound
        best_puct, best_action = -np.inf, None
        for a in range(self.game_rules.get_action_space()):
            if valid_actions[a]:
                puct = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * np.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])

                if puct > best_puct:
                    best_puct = puct
                    best_action = a

        # simulate the action that maximized ucb
        a = best_action
        new_board, _ = self.game_rules.step(board, a, 1)
        new_board = self.game_rules.perspective(new_board, -1)

        # keep simulating until we find a leaf node
        value = self.simulate(new_board)
        
        # update this states' values using the backpropagated data from the search
        if (s, a) in self.Qsa:
            self.Nsa[(s, a)] += 1
            self.Qsa[(s, a)] = ((self.Qsa[(s, a)] * (self.Nsa[(s, a)] - 1)) + value) / (self.Nsa[(s, a)])
        else:
            self.Nsa[(s, a)] = 1
            self.Qsa[(s, a)] = value
        
        self.Ns[s] += 1
        return -value