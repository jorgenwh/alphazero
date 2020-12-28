import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from collections import deque

from mcts import MCTS

class AlphaZero:
    def __init__(self, game_rules, nnet, args):
        self.game_rules = game_rules
        self.nnet = nnet
        self.args = args

        self.training_data = deque(maxlen=self.args.play_memory)

    def train(self):
        """
        The main training loop:
            - self-play args.episodes episodes and add the generated data to self.training_data
            - train the neural network on the generated (and previously stored) data
            - pit the network checkpointed before training against the newly trained network and save the updated network if it wins at least 55% of the games against the old network
        """
        for i in range(self.args.iterations):
            print(f"Iteration {i+1}/{self.args.iterations}")

            # self-play to generate training data
            mcts = MCTS(self.game_rules, self.nnet, self.args)
            for sp in tqdm(range(self.args.episodes), desc="Self-play"):
                examples = self.self_play(mcts)
                self.training_data.extend(examples)

            pass

    def self_play(self, mcts):
        examples = []
        board = self.game_rules.start_board()
        cur_player = 1
        step = 0

        while not self.game_rules.terminal(board):
            step += 1
            board_perspective = self.game_rules.perspective(board, cur_player)
            t = int(step < self.args.exploration_temp_threshold)

            pi = mcts.tree_search(board_perspective, t)

            # temp action chooser
            valid_moves = self.game_rules.get_valid_actions(board_perspective)
            valids = []
            for i in range(len(valid_moves)):
                if valid_moves[i]:
                    valids.append(i)
            action = np.random.choice(valids)
            # temp action chooser end

            #action = np.random.choice(self.game_rules.get_action_space(), p=pi)
            board, cur_player = self.game_rules.step(board, action, cur_player)

        result = self.game_rules.result(board, cur_player)
        return []