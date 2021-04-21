from tqdm import tqdm
import numpy as np

from mcts import MCTS

class Pit:
    """
    Pit class overseeing neural net evaluations
    """
    def __init__(self, game_rules, nnet, oppnnet, args):
        self.game_rules = game_rules
        self.nnet = nnet
        self.oppnnet = oppnnet
        self.args = args

    def evaluate(self):
        wins = 0
        losses = 0
        played = 0

        nnet = MCTS(self.game_rules, self.nnet, self.args)
        oppnnet = MCTS(self.game_rules, self.oppnnet, self.args)
        player = 1

        t = tqdm(range(self.args.eval_matches), desc="Evaluating")
        for _ in t:
            result = self.play_single_game(nnet, oppnnet) if player == 1 else self.play_single_game(oppnnet, nnet)
            played += 1
            wins += (player == 1 and result == 1) or (player == -1 and result == -1)
            losses += (player == 1 and result == -1) or (player == -1 and result == 1)
            t.set_postfix({"W/T/L": f"{wins}/{played - (wins + losses)}/{losses}"})

        return wins, played - (wins + losses), losses

    def play_single_game(self, mcts1, mcts2):
        """
        Play out a single match between two nnets
        """
        cur_player = 1
        board = self.game_rules.get_start_board()

        while not self.game_rules.terminal(board):
            board_perspective = self.game_rules.perspective(board, cur_player)

            if cur_player == 1:
                pi = mcts1.get_policy(board_perspective, t=0)
            else:
                pi = mcts2.get_policy(board_perspective, t=0)

            action = np.argmax(pi)
            board, cur_player = self.game_rules.step(board, action, cur_player)

        if cur_player == 1:
            board = self.game_rules.perspective(board, cur_player)
            return self.game_rules.result(board, cur_player)
        else:
            return self.game_rules.result(board, 1)
            