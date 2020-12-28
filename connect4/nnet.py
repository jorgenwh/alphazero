import torch
from tqdm import tqdm

from .connect4_network import Connect4_Model

class Connect4_Network:
    def __init__(self, game_rules, args):
        self.game_rules = game_rules
        self.args = args
        self.model = Connect4_Model(self.args)