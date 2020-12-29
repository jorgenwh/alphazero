import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from collections import deque

from connect4.nnet import Connect4_Network
from utils import setup_session, save_checkpoint, load_checkpoint
from mcts import MCTS

class AlphaZero:
    def __init__(self, game_rules, nnet, args):
        self.game_rules = game_rules
        self.args = args
        self.nnet = nnet
        self.oppnnet = Connect4_Network(self.game_rules, self.args)
        self.training_data = deque(maxlen=self.args.play_memory)

        self.sess_num = setup_session()
        self.checkpoint_num = 0
        save_checkpoint(self.nnet, self.sess_num, self.checkpoint_num)

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
            print(f"Self-play: {self.args.episodes} episodes")
            for sp in tqdm(range(self.args.episodes), desc="Self-play"):
                mcts = MCTS(self.game_rules, self.nnet, self.args)
                examples = self.self_play(mcts)
                self.training_data.extend(examples)

            # train the network
            print("Training neural network")
            self.nnet.train(self.training_data)

            # load the latest checkpoint into the opponent network
            load_checkpoint(self.oppnnet, self.sess_num, self.checkpoint_num)

            # pit the new and the previous networks against each other to evaluate the performance of the newly updated network
            print(f"Pit playoff: {int(self.args.playoff_episodes / 2)} matches as player 1 - {int(self.args.playoff_episodes / 2)} matches as player 2")
            wins, ties, losses = self.pit()
            if wins + losses == 0:
                score = 0
            else:
                score = wins / wins + losses
            print(f"wins: {wins} - ties: {ties} - losses: {losses} - score: {round(score, 3)}")
            if score >= self.args.playoff_threshold:
                print("Updated network accepted")
                self.checkpoint_num += 1
                save_checkpoint(self.nnet, self.sess_num, self.checkpoint_num)
            else:
                print("Updated network rejected")
                load_checkpoint(self.nnet, self.sess_num, self.checkpoint_num)

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
            examples.append((board_perspective, pi, cur_player))

            action = np.random.choice(self.game_rules.get_action_space(), p=pi)
            board, cur_player = self.game_rules.step(board, action, cur_player)

        result = self.game_rules.result(board, cur_player)
        examples = [(e[0], e[1], result * ((-1) ** (e[2] != cur_player))) for e in examples]
        return examples

    def pit(self):
        wins = 0
        ties = 0
        losses = 0

        for _ in tqdm(range(int(self.args.playoff_episodes / 2)), desc="Pit playoff (player1)"):
            result = self.playoff(
                MCTS(self.game_rules, self.nnet, self.args),
                MCTS(self.game_rules, self.oppnnet, self.args)
            )
            if result == 1:
                wins += 1
            elif result == 0:
                ties += 1
            else:
                losses += 1

        for _ in tqdm(range(int(self.args.playoff_episodes / 2)), desc="Pit playoff (player2)"):
            result = self.playoff(
                MCTS(self.game_rules, self.oppnnet, self.args),
                MCTS(self.game_rules, self.nnet, self.args)
            )
            if result == 1:
                losses += 1
            elif result == 0:
                ties += 1
            else:
                wins += 1

        return wins, ties, losses

    def playoff(self, mcts1, mcts2):
        cur_player = 1
        board = self.game_rules.start_board()

        while not self.game_rules.terminal(board):
            board_perspective = self.game_rules.perspective(board, cur_player)

            if cur_player == 1:
                pi = mcts1.tree_search(board_perspective, 0)
            else:
                pi = mcts2.tree_search(board_perspective, 0)

            action = np.argmax(pi)
            board, cur_player = self.game_rules.step(board, action, cur_player)

        if cur_player == 1:
            board = self.game_rules.perspective(board, cur_player)
            return self.game_rules.result(board, cur_player)
        else:
            return self.game_rules.result(board, 1)