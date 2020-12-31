import numpy as np
from tqdm import tqdm

from mcts import MCTS

class Self_Play:
    def __init__(self, game_rules, nnet, args):
        self.game_rules = game_rules
        self.nnet = nnet
        self.args = args
        self.training_data = []

    def play(self):
        for sp in tqdm(range(self.args.episodes), desc="Self-play"):
            mcts = MCTS(self.game_rules, self.nnet, self.args)
            self.play_game(mcts)
        return self.training_data

    def play_game(self, mcts):
        sequence = []
        board = self.game_rules.start_board()
        cur_player = 1
        ply = 0

        while not self.game_rules.terminal(board):
            board_perspective = self.game_rules.perspective(board, cur_player)
            temperature = int(ply + 1 < self.args.exploration_temp_threshold)
            
            mcts.tree_search(board_perspective)
            pi = mcts.get_policy(board_perspective, temperature)

            equal_positions = self.game_rules.get_equal_positions(board_perspective, pi)
            for board_, pi_ in equal_positions:
                sequence.append((board_, pi_, cur_player))

            action = np.random.choice(self.game_rules.get_action_space(), p=pi)
            board, cur_player = self.game_rules.step(board, action, cur_player)

            ply += 1

        value = self.game_rules.result(board, 1)
        for board, pi, player in sequence:
            v = (1 if player == value else -1) if value else 0
            self.training_data.append((board, pi, v))