import os
import sys
import time
import torch
import numpy as np
from tqdm import tqdm
from collections import deque

from alphazero.selfplay import selfplay
from alphazero.evaluate import evaluate
from alphazero.rules import Rules
from alphazero.network import Network
from alphazero.mcts import MCTS
from alphazero.misc import Arguments, session_setup, save_checkpoint, load_checkpoint, get_time_stamp

class Manager():
    """
    AlphaZero manager class running the training pipeline.
    """
    def __init__(self, rules: Rules, network: Network, args: Arguments):
        self.rules = rules
        self.network = network
        self.args = args

        self.checkpoint_network = self.network.__class__(self.args)
        
        self.session_number = session_setup(self.rules, self.args)
        self.checkpoint_number = 0
        save_checkpoint(self.network, self.session_number, self.checkpoint_number, self.args)
        load_checkpoint(self.network, self.session_number, self.checkpoint_number, self.args)

        self.training_examples = deque(maxlen=self.args.replay_memory)

    def train(self) -> None:
        """
        Iterate the training pipeline for [arguments.iterations] iterations.
        """
        for i in range(self.args.iterations):
            print(f"Iteration: {i + 1}/{self.args.iterations}")
            self.iterate()
    
    def iterate(self) -> None:
        """
        Performs a single iteration of the training pipeline.
        """
        st = time.time()
        print(f"Self-Play: ({self.args.episodes} episodes)")
        training_examples = selfplay(self.rules, self.network, self.args)
        self.training_examples.extend(training_examples)

        # Train the network
        print(f"\nTraining Neural Network")
        self.network.train(self.training_examples)

        # Pit the new and previous network against each other to evaluate updated performance.
        # Only if the updated network has improved significantly will it be saved as the next checkpoint
        print(f"\nPit Evaluation")
        wins, ties, losses = evaluate(self.rules, self.network, self.checkpoint_network, self.args)
        
        score = wins / max((wins + losses), 1)
        print(f"W: {wins} - T: {ties} - L: {losses} - Score: {round(score, 3)}")

        if score >= self.args.acceptance_threshold:
            print("Checkpoint Accepted")
            self.checkpoint_number += 1
            save_checkpoint(self.network, self.session_number, self.checkpoint_number, self.args)
            load_checkpoint(self.checkpoint_network, self.session_number, self.checkpoint_number, self.args)
        else:
            print("Checkpoint Discarded")
            load_checkpoint(self.network, self.session_number, self.checkpoint_number, self.args)

        t = get_time_stamp(time.time() - st)
        print(f"Iteration time: {t}")
        print(f"Checckpoint: {self.checkpoint_number}\n\n")

