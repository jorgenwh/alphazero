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

    def tree_search(self, board, t):
        # run monte carlo tree search simulations
        for _ in range(self.args.monte_carlo_simulations):
            self.simulate(board)

        s = self.game_rules.tostring(board)
        sims = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game_rules.get_action_space())]

        if t > 0:
            # if temperature is above 0 (we don't return a greedy policy)
            probs = [n ** (1 / t) for n in sims]
            p_sum = np.sum(probs)
            pi = np.array([p / p_sum for p in probs])
            return pi
        else:
            # if t = 0 we return a greedy policy
            pi = np.zeros(self.game_rules.get_action_space())
            best_actions = []
            for a in range(self.game_rules.get_action_space()):
                if sims[a] == np.max(sims):
                    best_actions.append(a)
            action = np.random.choice(best_actions)
            pi[action] = 1
            return pi

    def simulate(self, board):
        """
        Input board will always be of the perspective of player 1
        """
        # if the game state is terminal
        if self.game_rules.terminal(board):
            value = self.game_rules.result(board, 1)
            return -value

        s = self.game_rules.tostring(board)
        valid_actions = self.game_rules.get_valid_actions(board)

        # if we haven't evaluated this position before (we have reached a leaf node)
        if s not in self.Ps:
            pi, value = self.nnet.evaluate(board, valid_actions)
            self.Ps[s] = pi
            return -value

        # find the action that maximizes the upper confidence bound
        best_ucb, best_action = -np.inf, None
        for a in range(self.game_rules.get_action_space()):
            if valid_actions[a]:
                ucb = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * np.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])

                if ucb > best_ucb:
                    best_ucb = ucb
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