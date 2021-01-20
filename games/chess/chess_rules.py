import chess
import numpy as np
from rules import Rules

class ChessRules(Rules):
    def __init__(self):
        pass

    def step(self, board, action, player):
        assert self.get_valid_actions(board, player)[action]
        pass

    def get_action_space(self):
        pass

    def get_valid_actions(self, board, player):
        pass

    def start_board(self):
        board = np.zeros((8, 8, 6))

        # place pawns
        board[1,:,0] = -1
        board[6,:,0] = 1

        # place knights
        board[0,1,1] = board[0,6,1] = -1
        board[7,1,1] = board[7,6,1] = 1

        # place bishops
        board[0,2,2] = board[0,5,2] = -1
        board[7,2,2] = board[7,5,2] = 1

        # place rooks
        board[0,0,3] = board[0,7,3] = -1
        board[7,0,3] = board[7,7,3] = 1

        # place kings and queens
        board[0,3,4] = board[0,4,5] = -1
        board[7,3,4] = board[7,4,5] = 1

        return board
