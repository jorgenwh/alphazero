import numpy as np

from rules import Rules

class Connect4_Rules(Rules):
    def __init__(self):
        pass

    def step(self, board, action, player):
        assert self.get_valid_actions(board)[action]
        row = self.lowest_row(board, action)
        new_board = board.copy()
        new_board[row,action] = player
        return new_board, -player

    def get_action_space(self):
        return 7

    def lowest_row(self, board, action):
        for row in range(5, -1, -1):
            if not board[row,action]:
                return row

    def get_valid_actions(self, board):
        valid_actions = [0] * self.get_action_space()
        for a in range(self.get_action_space()):
            if not board[0,a]:
                valid_actions[a] = 1
        return valid_actions

    def start_board(self):
        return np.zeros((6, 7))

    def perspective(self, board, player):
        return board * player

    def get_equal_positions(self, board, pi):
        positions = [(board, pi)]
        positions.append((np.flip(board, 1), np.flip(pi)))
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
        for c in range(4):
            for r in range(6):
                if board[r,c] == board[r,c+1] == board[r,c+2] == board[r,c+3] == player:
                    return True

        for c in range(7):
            for r in range(3):
                if board[r,c] == board[r+1,c] == board[r+2,c] == board[r+3,c] == player:
                    return True

        for c in range(4):
            for r in range(3):
                if board[r,c] == board[r+1,c+1] == board[r+2,c+2] == board[r+3,c+3] == player:
                    return True

        for c in range(4):
            for r in range(3, 6):
                if board[r,c] == board[r-1,c+1] == board[r-2,c+2] == board[r-3,c+3] == player:
                    return True

        return False