import numpy as np

from rules import Rules

class TicTacToe_Rules(Rules):
    def __init__(self):
        pass

    def step(self, board, action, player):
        assert self.get_valid_actions(board)[action]
        r = int(action/3)
        c = action % 3
        new_board = board.copy()
        new_board[r,c] = player
        return new_board, -player

    def get_action_space(self):
        return 9

    def get_valid_actions(self, board):
        valid_actions = [0] * self.get_action_space()
        for r in range(3):
            for c in range(3):
                if not board[r,c]:
                    valid_actions[r*3 + c] = 1
        return valid_actions

    def start_board(self):
        return np.zeros((3, 3))

    def perspective(self, board, player):
        return board * player

    def get_equal_positions(self, board, pi):
        # TODO rotate board to get more equal positions
        positions = [(board, pi)]

        positions.append((np.flip(board, 0), np.flip(np.array(pi).reshape(board.shape), 0).flatten()))
        positions.append((np.flip(board, 1), np.flip(np.array(pi).reshape(board.shape), 1).flatten()))

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
        for r in range(3):
            if board[r,0] == board[r,1] == board[r,2] == player:
                return True
        
        for c in range(3):
            if board[0,c] == board[1,c] == board[2,c] == player:
                return True
        
        if board[0,0] == board[1,1] == board[2,2] == player:
            return True
        
        if board[2,0] == board[1,1] == board[0,2] == player:
            return True

        return False