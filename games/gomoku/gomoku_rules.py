import numpy as np

from rules import Rules

class Gomoku_Rules(Rules):
    def __init__(self, size):
        self.size = size

    def step(self, board, action, player):
        assert self.get_valid_actions(board)[action]
        row = int(action / self.size)
        col = action % self.size
        new_board = board.copy()
        new_board[row,col] = player
        return new_board, -player

    def get_action_space(self):
        return self.size ** 2

    def get_valid_actions(self, board):
        valid_actions = [0] * self.get_action_space()
        for r in range(self.size):
            for c in range(self.size):
                if not board[r,c]:
                    valid_actions[r*self.size + c] = 1
        return valid_actions

    def start_board(self):
        return np.zeros((self.size, self.size))

    def perspective(self, board, player):
        return board * player

    def get_equal_positions(self, board, pi):
        positions = [(board, pi)]
        """p = np.array(pi).reshape(board.shape)
        
        for i in range(3):
            b, p = np.rot90(board), np.rot90(p)
            positions.append((b, p.copy().flatten()))

        positions.append((np.flip(board, 0), np.flip(p, 0).flatten()))
        positions.append((np.flip(board, 1), np.flip(p, 1).flatten()))"""

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
        for c in range(self.size-4):
            for r in range(self.size):
                if board[r,c] == board[r,c+1] == board[r,c+2] == board[r,c+3] == board[r,c+4] == player:
                    return True

        for c in range(self.size):
            for r in range(self.size-4):
                if board[r,c] == board[r+1,c] == board[r+2,c] == board[r+3,c] == board[r+4,c] == player:
                    return True

        for c in range(self.size-4):
            for r in range(self.size-4):
                if board[r,c] == board[r+1,c+1] == board[r+2,c+2] == board[r+3,c+3] == board[r+4,c+4] == player:
                    return True

        for c in range(self.size-4):
            for r in range(4, self.size):
                if board[r,c] == board[r-1,c+1] == board[r-2,c+2] == board[r-3,c+3] == board[r-4,c+4] == player:
                    return True

        return False