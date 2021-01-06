import numpy as np
from tqdm import tqdm

from mcts import MCTS

class Self_Play:
    """
    Self-play class.
    """
    def __init__(self, game_rules, nnet, args):
        self.game_rules = game_rules
        self.nnet = nnet
        self.args = args
        self.training_data = []

    def play(self):
        """
        Performs args.episodes games of self-play using the neural network before returning the generated
        game data.

        A MCTS is initiated at the beginning of each game, and remains until the end of the game,
        opposed to creating a new MCTS for each move.
        """
        for sp in tqdm(range(self.args.episodes), desc="Self-play"):
            mcts = MCTS(self.game_rules, self.nnet, self.args)
            self.play_game(mcts)
        return self.training_data

    def play_game(self, mcts):
        """
        Performs a single episode of self-play using the provided MCTS object.
        Relevant game data is stored for each action taken during the episode. At the end of the episode
        the actual training data is created by adding the winner/loser of the game.
        """
        exploration_temp_threshold = np.random.randint(4, 60)
        sequence = []
        board = self.game_rules.start_board()
        cur_player = 1
        ply = 0

        while not self.game_rules.terminal(board):
            board_perspective = self.game_rules.perspective(board, cur_player)
            
            mcts.tree_search(board_perspective)
            pi = mcts.get_policy(board_perspective, int(ply + 1 < exploration_temp_threshold))

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
            