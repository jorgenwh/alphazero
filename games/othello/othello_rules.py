import numpy as np

from rules import Rules

class Othello_Rules(Rules):
    def __init__(self):
        pass

    def step(self, board, action, player):
        assert self.get_valid_actions(board)[action]
        r = int(action / 8)
        c = action % 8
        new_board = board.copy()
        new_board[r,c] = player
        return new_board, -player

    def get_action_space(self):
        return 8 ** 2

    def get_valid_actions(self, board):
        valid_actions = [0] * self.get_action_space()
        valid_set = set()
        for a in range(self.get_action_space()):
            valid_set.update(self.get_valid_set(board, a))
        for a in valid_set:
            valid_actions[a] = 1
        
        return valid_actions

    def get_valid_set(self, board, a):
        moves = []
        x, y = int(a / 8), a % 8
        for direction in [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]:
            cur_x, cur_y = x + direction[0], y + direction[1]
            

    def start_board(self):
        board = np.zeros((8, 8))
        board[3,3] = board[4,4] = -1
        board[3,4] = board[4,3] = 1

    def perspective(self, board, player):
        return board * player

    def get_equal_positions(self, board, pi):
        positions = [(board, pi)]
        # TODO
        return positions
        
    def tostring(self, board):
        return board.tostring()

    def terminal(self, board):
        return sum(self.get_valid_actions(board)) == 0 or self.is_winner(board, 1) or self.is_winner(board, -1)

    def result(self, board, player):
        if self.is_winner(board, player):
            return 1
        elif self.is_winner(board, -player):
            return -1
        else:
            return 0

    def is_winner(self, board, player):
        score = np.sum(board)
        if player == 1:
            return score > 0
        else:
            return score < 0