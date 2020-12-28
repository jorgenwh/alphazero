import os

from utils import Dotdict

from connect4.connect4_rules import Connect4_Rules as Rules
from connect4.nnet import Connect4_Network
from alphazero import AlphaZero

args = Dotdict({
    # alphazero
    "iterations": 10,
    "episodes": 100,
    "play_memory": 150_000,
    "exploration_temp_threshold": 14, 
    "playoff_threshold": 0.55,
    "cpuct": 1,

    # monte carlo tree search
    "monte_carlo_simulations": 50,

    # neural network
    "lr": 0.001,
    "num_channels": 512,
    "batch_size": 64,
    "cuda": True
})

if __name__ == "__main__":
    game_rules = Rules()
    nnet = Connect4_Network(game_rules, args)
    alphazero = AlphaZero(game_rules, nnet, args)
    alphazero.train()
