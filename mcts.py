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

    def get_policy(self, board, t):
        s = self.game_rules.tostring(board)
        raw_pi = [self.Nsa[(s, a)] for a in range(self.game_rules.get_action_space())]

        if t > 0:
            # choose stochastically
            pi = [N ** (1 / t) for N in raw_pi]
            pi = [N / sum(pi) for N in pi]
        else:
            # choose move deterministically
            actions = [i for i in range(len(raw_pi)) if raw_pi[i] == max(raw_pi)]
            pi = [0] * len(raw_pi)
            pi[np.random.choice(actions)] = 1
            
        return pi

    def tree_search(self, board):
        # run monte carlo tree search simulations
        for _ in range(self.args.monte_carlo_simulations):
            self.simulate(board)

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
        puct, action = -np.inf, None
        for a in range(self.game_rules.get_action_space()):
            if valid_actions[a]:
                puct_ = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * np.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])

                if puct_ > puct:
                    puct = puct_
                    action = a

        # simulate the chosen action
        new_board, _ = self.game_rules.step(board, action, 1)
        new_board = self.game_rules.perspective(new_board, -1)

        # keep simulating until we find a leaf node
        value = self.simulate(new_board)
        
        # update this states' q value using the backpropagated data from the search
        self.Nsa[(s, action)] += 1
        self.Qsa[(s, action)] = ((self.Qsa[(s, action)] * (self.Nsa[(s, action)] - 1)) + value) / (self.Nsa[(s, action)])
        self.Ns[s] += 1

        return -value