from tqdm import tqdm
import numpy as np

from mcts import MCTS

class Pit:
    """
    The neural network evaluation pit class, handling the evaluation
    of updated neural networks to determine whether they should be checkpointed or deleted.
    """
    def __init__(self, game_rules, nnet, oppnnet, args):
        self.game_rules = game_rules
        self.nnet = nnet
        self.oppnnet = oppnnet
        self.args = args

    def evaluate(self):
        """
        Performs the evaluation by playing args.eval_matches matches between the updated- and the
        previously checkpointed networks.
        """
        wins = 0
        ties = 0
        losses = 0

        nnet = MCTS(self.game_rules, self.nnet, self.args)
        oppnnet = MCTS(self.game_rules, self.oppnnet, self.args)
        player = 1

        t = tqdm(range(self.args.eval_matches), desc="Evaluating")
        for _ in t:
            result = self.match(nnet, oppnnet) if player == 1 else self.match(oppnnet, nnet)
            if player == 1:
                if result == 1:
                    wins += 1
                elif result == -1:
                    losses += 1
                else:
                    ties += 1
                player = -1
            else:
                if result == 1:
                    losses += 1
                elif result == -1:
                    wins += 1
                else:
                    ties += 1
                player = 1

            t.set_postfix({"W/T/L": f"{wins}/{ties}/{losses}"})

        return wins, ties, losses

    def match(self, mcts1, mcts2):
        """
        Perform a single match between two networks.
        """
        cur_player = 1
        board = self.game_rules.start_board()

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
            