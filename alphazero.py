import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from collections import deque

from self_play import Self_Play
from pit import Pit
from utils import setup_session, save_checkpoint, load_checkpoint, save_model

class AlphaZero:
    """
    The main AlphaZero class performing the overall training pipeline.
    """
    def __init__(self, game_rules, nnet, args, nn_class):
        self.game_rules = game_rules
        self.nnet = nnet
        self.args = args

        self.oppnnet = nn_class(self.game_rules, self.args)
        self.sess_num = setup_session()
        self.checkpoint_num = 0
        save_checkpoint(self.nnet, self.sess_num, self.checkpoint_num)
        load_checkpoint(self.oppnnet, self.sess_num, self.checkpoint_num)
        self.training_data = deque(maxlen=self.args.play_memory)

    def train(self):
        """
        Train AlphaZero for args.iterations iterations before finally saving the resulting
        model after all the training steps are completed.

        If training is terminated before the final model is saved, the latest checkpoint model under
        sessions/session(n)/ will contain the same weights as the latest best model.
        """
        for i in range(self.args.iterations):
            print(f"Iteration {i+1}/{self.args.iterations}")
            self.iterate()

        save_model(self.nnet, self.args.game + "_model")

    def iterate(self):
        """
        Perform one training iteration:
            - Self-play for args.episodes games to generate training data.
            - Train the neural network on the generated training data (and previously generated 
                training data).
            - Evaluate the updated network against the latest checkpoint to determine whether to 
                checkpoint the updated network or delete it.
        """
        print(f"\nSelf-play: ({self.args.episodes} episodes)")
        self_play = Self_Play(self.game_rules, self.nnet, self.args)
        training_data = self_play.play()
        self.training_data.extend(training_data)

        # train the network
        print("\nTraining Neural Network")
        self.nnet.train(self.training_data)

        # pit the new and the previous networks against each other to evaluate the performance of the newly updated network
        """print(f"\nPit Evaluation")
        pit = Pit(self.game_rules, self.nnet, self.oppnnet, self.args)
        wins, ties, losses = pit.evaluate()

        score = wins / max((wins + losses), 1)
        print(f"W: {wins} - T: {ties} - L: {losses} - Score: {round(score, 3)}")

        if score >= self.args.eval_score_threshold:
            print("Updated Network Accepted")
            self.checkpoint_num += 1
            save_checkpoint(self.nnet, self.sess_num, self.checkpoint_num)
            load_checkpoint(self.oppnnet, self.sess_num, self.checkpoint_num)
        else:
            print("Updated Network Rejected")
            load_checkpoint(self.nnet, self.sess_num, self.checkpoint_num)

        print(f"Checkpoint: {self.checkpoint_num}\n\n")"""

        self.checkpoint_num += 1
        save_checkpoint(self.nnet, self.sess_num, self.checkpoint_num)
        load_checkpoint(self.oppnnet, self.sess_num, self.checkpoint_num)
        