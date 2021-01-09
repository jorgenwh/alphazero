import numpy as np
from collections import defaultdict

class MCTS:
    """
    Monte-Carlo tree search class.
    The implementation of the MCTS is mimicked from the original AlphaGo Zero paper from DeepMind. 
    """
    def __init__(self, game_rules, nnet, args):
        self.game_rules = game_rules
        self.nnet = nnet
        self.args = args
        
        self.N = defaultdict(float)
        self.W = defaultdict(float)
        self.Q = defaultdict(float)
        self.P = defaultdict(float)

    def get_policy(self, board, t):
        self.tree_search(board)

        s = self.game_rules.tostring(board)
        raw_pi = [self.N[(s, a)] for a in range(self.game_rules.get_action_space())]
        
        """
        If t == 0, we choose the move deterministically (for competitive play). According to the original
        AlphaGo Zero paper this was done in the same way as for t > 0, but with an infinitesimal value for t.
        This caused too large values in some cases, so instead an action is chosen randomly from all actions 
        a = argmax(pi).
        """
        if t > 0:
            pi = [N ** (1 / t) for N in raw_pi]
            if sum(pi) == 0:
                pi = [1 for _ in range(len(pi))]
            pi = [N / sum(pi) for N in pi]
        else:
            actions = [i for i in range(len(raw_pi)) if raw_pi[i] == max(raw_pi)]
            pi = [0] * len(raw_pi)
            pi[np.random.choice(actions)] = 1
            
        return pi

    def tree_search(self, board):
        """
        Runs the monte carlo simulations.
        """
        for _ in range(self.args.monte_carlo_simulations):
            self.simulate(board)

    def simulate(self, board):
        """
        If the rollout has reached a terminal game state, it returns the position value provided by 
        the game's rules. 
        The returned value is negative of the value for player 1, because the positive 
        value for the current player will the equivalent to the negative value for the previous player.
        """
        if self.game_rules.terminal(board):
            value = self.game_rules.result(board, 1)
            return -value

        s = self.game_rules.tostring(board)
        valid_actions = self.game_rules.get_valid_actions(board, 1)

        """
        If we have reached a leaf node (unevaluated position node), we evaluate the position and store the
        action probabilities so that we don't need to pass the position through the network again for future rollouts.
        The action probabilities for the position is masked to discard any invalid actions.
        Finally, the negative of the value evaluation from the network is returned.
        """
        if s not in self.P:
            pi, value = self.nnet.evaluate(board)

            # Mask the invalid actions from the action probability tensor before
            # renormalizing the output probabilities.
            
            # TODO Fix an issue where the NN only assigns value to invalid moves, resulting in the sum of the policy
            # to be 0. This currently leaves the saved policy for the state to be 0 for all moves.
            pi = pi * valid_actions
            pi = pi / max(np.sum(pi), 1e-8)

            self.P[s] = pi
            return -value

        """
        Here we select the action to continue the simulation.
        The action chosen is whatever action maximizes Q(s, a) + U(s, a) in accordance with DeepMind's
        AlphaGo Zero paper.

        a(t) = argmax(Q(s, a) + U(s, a)), where
        U(s, a) = cpuct * P(s, a) * (sprt(N(s)) / (1 + N(s, a)))

        'cpuct' is a constant determining the level of exploration. Low cpuct values will give more weight to
        the Q value of the action compared to the network's output probabilities and the amount of times the action
        has been explored.
        """
        QU, action = -np.inf, None
        for a in range(self.game_rules.get_action_space()):
            # Only calculate the QU value for valid actions
            if valid_actions[a]:
                Q = self.Q[(s, a)]
                U = self.args.cpuct * self.P[s][a] * np.sqrt(sum([self.N[(s, a_)] for a_ in range(self.game_rules.get_action_space())])) / (1 + self.N[(s, a)])
                QU_a = Q + U

                if QU_a > QU:
                    QU = QU_a
                    action = a

        # Simulate the action and get the next position node.
        # Convert the next node to the perspective of the current player for that position.
        new_board, _ = self.game_rules.step(board, action, 1)
        new_board = self.game_rules.perspective(new_board, -1)

        # Get the value from the continued search.
        value = self.simulate(new_board)
        
        # Finally, update the statistics for the current position node.
        self.N[(s, action)] += 1
        self.W[(s, action)] += value
        self.Q[(s, action)] = self.W[(s, action)] / self.N[(s, action)]

        return -value
