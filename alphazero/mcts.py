import numpy as np
from collections import defaultdict

from .rules import Rules
from .network import Network
from .config import Config

CPUCT = 1.0

class MCTS:
    def __init__(self, rules: Rules, network: Network, config: Config):
        self.rules = rules
        self.network = network
        self.config = config

        self.N = defaultdict(float) # Number of times a state-action pair has been visited
        self.W = defaultdict(float) # Total value of a state-action pair
        self.Q = defaultdict(float) # Average value of a state-action pair
        self.P = defaultdict(float) # Prior probability of taking an action in a state

    def get_policy(self, state: np.ndarray) -> np.ndarray:
        for _ in range(self.config.MONTE_CARLO_ROLLOUTS):
            self.leaf_rollout(state)

        state_hash = self.rules.hash(state)
        raw_pi = [self.N[(state_hash, action)] for action in range(self.rules.get_action_space())]

        # if temperature = 0, we choose the action deterministically for competitive play
        if self.config.TEMPERATURE > 0:
            pi = [N ** (1.0 / self.config.TEMPERATURE) for N in raw_pi]
            if sum(pi) == 0:
                pi = [1.0 for _ in range(len(pi))]
            pi = [N / sum(pi) for N in pi]
        else:
            max_action = max(raw_pi)
            pi = [0] * len(self.rules.get_action_space())
            for action in self.rules.get_action_space():
                if raw_pi[action] == max_action:
                    pi[action] = 1
                    break

        return pi

    def leaf_rollout(self, state: np.ndarray) -> float:
        # if the search has reached a terminal state, it returns the value according to
        # the game's rules
        winner = self.rules.get_winner(state)
        if winner is not None:
            return -winner

        state_hash = self.rules.hash(state)
        valid_actions = self.rules.get_valid_actions(state, 1)

        # if a leaf node is reached, we evaluate using the network
        if state_hash not in self.P:
            pi, v = self.network(state)

            # mask invalid actions from the action probabilities and renormalize
            pi = pi * valid_actions
            pi = pi / max(np.sum(pi), 1e-8)

            self.P[state_hash] = pi
            return -v

        # we select the action used to continue the search by choosing the action that 
        # maximizes Q(s, a) + U(s, a) in accordance with DeepMind's AlphaGo Zero paper:
        #
        #   a(t) = argmax(Q(s, a) + U(s, a)), where
        #   U(s, a) = cpuct * P(s, a) * (sprt(N(s)) / (1 + N(s, a)))
        #
        # 'cpuct' is a constant determining the level of exploration - a small 
        # cpuct value will give more weight to the action Q-value as opposed to the network's 
        # output probabilities and the amount of times the action has been explored

        qu, selected_action = -np.inf, None
        for action in range(self.rules.get_action_space()):
            if valid_actions[action] == 0:
                continue

            q = self.Q[(state_hash, action)]
            u = CPUCT * self.P[state_hash][action] * np.sqrt(sum([self.N[(state_hash, a)] for a in range(self.rules.get_action_space())])) / (1.0 + self.N[(state_hash, action)])
            qu_a = q + u

            if qu_a > qu:
                qu = qu_a
                selected_action = action

        # simulate the action and get the next positional node
        next_state = self.rules.step(state, selected_action, 1)
        next_state = self.rules.flip_view(next_state)

        # continue search
        v = self.leaf_rollout(next_state)

        self.N[(state_hash, selected_action)] += 1
        self.W[(state_hash, selected_action)] += v
        self.Q[(state_hash, selected_action)] = self.W[(state_hash, selected_action)] / self.N[(state_hash, selected_action)]

        return -v

