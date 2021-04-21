import numpy as np
from tqdm import tqdm

from mcts import MCTS

class Self_Play:
    """
    Class to oversee self-play
    """
    def __init__(self, game_rules, nnet, args):
        self.game_rules = game_rules
        self.nnet = nnet
        self.args = args
        self.training_data = []

    def play(self):
        for sp in tqdm(range(self.args.episodes), desc="Self-play"):
            mcts = MCTS(self.game_rules, self.nnet, self.args)
            self.play_single_game(mcts)

        return self.training_data

    def play_single_game(self, mcts):
        sequence = []
        board = self.game_rules.get_start_board()
        cur_player = 1
        ply = 0

        while not self.game_rules.terminal(board):
            board_perspective = self.game_rules.perspective(board, cur_player)
            pi = mcts.get_policy(board_perspective, self.args.temperature)

            sequence.append((board_perspective, pi, cur_player))

            action = np.random.choice(self.game_rules.get_action_space(), p=pi)
            board, cur_player = self.game_rules.step(board, action, cur_player)
            ply += 1
        
        value = self.game_rules.result(board, 1)
        for board, pi, player in sequence:
            v = (1 if player == value else -1) if value else 0
            self.training_data.append((board, pi, v))
            