import numpy as np

from games.connect4.connect4_rules import Connect4Rules
from games.othello.othello_rules import OthelloRules
from games.tictactoe.tictactoe_rules import TicTacToeRules
from games.gomoku.gomoku_rules import GomokuRules

class Minimax:
    """
    Minimax tree search class.
    """
    def __init__(self, game_rules, args):
        self.game_rules = game_rules
        self.args = args

    def get_policy(self, board, t):
        """
        Input:
            board (np.array): the board from the current player's perspective.
            t (float): the exploration temperature.
        """
        valid_actions = self.game_rules.get_valid_actions(board, 1)
        pi = [None for _ in valid_actions]

        for a in range(len(valid_actions)):
            if valid_actions[a]:
                next_board, _ = self.game_rules.step(board, a, 1)
                v = self.mini(next_board, self.args.minimax, -np.inf, np.inf)
                pi[a] = v
            else:
                pi[a] = -np.inf
        
        return pi

    def maxi(self, board, depth, alpha, beta):
        if self.game_rules.terminal(board) or depth == 0:
            return self.evaluate(board)

        value = -np.inf
        valid_actions = self.game_rules.get_valid_actions(board, 1)

        if not sum(valid_actions):
            next_board, _ = self.game_rules.step(board, None, 1)
            return self.mini(next_board, depth - 1, alpha, beta)

        for a in range(len(valid_actions)):
            if valid_actions[a]:
                next_board, _ = self.game_rules.step(board, a, 1)
                v = self.mini(next_board, depth - 1, alpha, beta)
                value = max(value, v)
                if alpha >= beta:
                    break

        return value

    def mini(self, board, depth, alpha, beta):
        if self.game_rules.terminal(board) or depth == 0:
            return self.evaluate(board)
        
        value = np.inf
        valid_actions = self.game_rules.get_valid_actions(board, -1)

        if not sum(valid_actions):
            next_board, _ = self.game_rules.step(board, None, -1)
            return self.maxi(next_board, depth - 1, alpha, beta)

        for a in range(len(valid_actions)):
            if valid_actions[a]:
                next_board, _ = self.game_rules.step(board, a, -1)
                v = self.maxi(next_board, depth - 1, alpha, beta)
                value = min(value, v)
                if alpha >= beta:
                    break

        return value

    def evaluate(self, board):
        if self.game_rules.terminal(board):
            return self.game_rules.result(board, 1) * 1000

        if isinstance(self.game_rules, Connect4_Rules):
            return self.evaluate_connect4(board)
        elif isinstance(self.game_rules, Othello_Rules):
            return self.evaluate_othello(board)
        elif isinstance(self.game_rules, TicTacToe_Rules):
            return self.evaluate_tictactoe(board)
        elif isinstance(self.game_rules, Gomoku_Rules):
            return self.evaluate_gomoku(board)
        else:
            raise TypeError("Invalid game rules type.")

    def evaluate_connect4(self, board):
        """
        Heuristic evaluation of a board for connect4. Evaluate the position from the perspective of player 1.
        """
        return 0

    def evaluate_othello(self, board):
        """
        Heuristic evaluation of a board for othello. Evaluate the position from the perspective of player 1.
        """
        return np.sum(board)

    def evaluate_tictactoe(self, board):
        """
        Heuristic evaluation of a board for tictactoe. Evaluate the position from the perspective of player 1.
        """
        return 0

    def evaluate_gomoku(self, board):
        """
        Heuristic evaluation of a board for gomoku. Evaluate the position from the perspective of player 1.
        """
        return 0
