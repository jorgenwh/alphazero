import os

from alphazero import AlphaZero
from connect4.connect4_rules import Connect4_Rules as Rules
from connect4.connect4_network import Connect4_Network
from utils import Dotdict

args = Dotdict({
    # AlphaZero
    "iterations": 1000, #how many update steps to perform
    "episodes": 100,
    "play_memory": 150_000, # how many training examples to store at a given time
    "exploration_temp_threshold": 15, 
    "playoff_score_threshold": 0.55,
    "playoffs": 40,
    "cpuct": 1, # controls exploration during self-play
    "monte_carlo_simulations": 25,

    # Neural Network
    "lr": 0.001,
    "epochs": 10,
    "batch_size": 64,
    "cuda": True
})

if __name__ == "__main__":
    game_rules = Rules()
    nnet = Connect4_Network(game_rules, args)
    alphazero = AlphaZero(game_rules, nnet, args)
    alphazero.iterate()
