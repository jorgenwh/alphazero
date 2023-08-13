import os
import time
import torch
import numpy as np
from collections import deque

from .rules import Rules
from .network import Network
from .replay_memory import ReplayMemory
from .selfplay import selfplay
from .evaluate import evaluate
from .args import ITERATIONS, EPISODES, REPLAY_MEMORY_SIZE, ACCEPTANCE_THRESHOLD
from .misc import PrintColors as PC, setup_training_session, get_time_stamp

def save_checkpoint(dir_name: str, network: Network, games_played: int) -> None:
    filename = f"{dir_name}/model_checkpoint_{games_played}games.pt"
    if os.path.isfile(filename):
        return
    torch.save(network.model.state_dict(), filename)

def load_checkpoint(dir_name: str, network: Network, games_played: int) -> None:
    filename = f"{dir_name}/model_checkpoint_{games_played}games.pt"
    assert os.path.isfile(filename)
    network.model.load_state_dict(torch.load(filename))


class Trainer():
    def __init__(self, rules: Rules, network: Network):
        self.dir_name = setup_training_session()     
        self.rules = rules
        self.network = network
        self.checkpoint_network = network.__class__()
        self.replay_memory = ReplayMemory(
                REPLAY_MEMORY_SIZE, self.rules.get_state_shape(), self.rules.get_action_space())
        self.played_games = 0
        self.last_saved_checkpoint = 0
        save_checkpoint(self.dir_name, self.network, self.played_games)
        load_checkpoint(self.dir_name, self.checkpoint_network, self.played_games)

    def start(self) -> None:
        for i in range(ITERATIONS):
            print(f"\n{PC.transparent}-----{PC.endc} {PC.bold}Iteration: {i + 1}/{ITERATIONS}{PC.endc} {PC.transparent}-----{PC.endc}\n")
            t1 = time.time()

            # self-play for training data generation
            print(f"{PC.yellow}self-play data generation{PC.endc}")
            self.played_games += selfplay(self.rules, self.network, self.replay_memory)

            # train network
            print(f"\n{PC.yellow}training neural network{PC.endc}")
            self.network.train(self.replay_memory)

            # pit trained network against the previous network to assert increase in strength
            print(f"\n{PC.yellow}evaluation{PC.endc}")
            wins, ties, losses = evaluate(self.rules, self.network, self.checkpoint_network)

            score = wins / max((wins + losses), 1)
            score_color = PC.green if score > ACCEPTANCE_THRESHOLD else PC.red
            print(f"wins: {PC.bold}{PC.green}{wins}{PC.endc} - ties: {PC.bold}{ties}{PC.endc} - losses: {PC.bold}{PC.red}{losses}{PC.endc} - score: {score_color}{round(score, 3)}{PC.endc}")

            if score > ACCEPTANCE_THRESHOLD:
                print(f"new network {PC.green}accepted{PC.endc} - saving checkpoint")
                self.last_saved_checkpoint = self.played_games
                save_checkpoint(self.dir_name, self.network, self.played_games)
                load_checkpoint(self.dir_name, self.checkpoint_network, self.played_games)
            else:
                print(f"checkpoint {PC.red}rejected{PC.endc} - discarding checkpoint")
                load_checkpoint(self.dir_name, self.network, self.last_saved_checkpoint)

            t2 = time.time()
            t = get_time_stamp(t2 - t1)
            print(f"\niteration time: {t}\n")
    
