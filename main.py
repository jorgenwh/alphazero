import os

from utils import Dotdict

from connect4.connect4_rules import Connect4_Rules as Rules
from connect4.nnet import Connect4_Network

args = Dotdict({
    # alphazero
    

    # neural network
    "num_channels": 512,
    "cuda": True
})

if __name__ == "__main__":
    game_rules = Rules()
    nnet = Connect4_Network(game_rules, args)

