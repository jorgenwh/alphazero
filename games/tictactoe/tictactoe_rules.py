import numpy as np
from rules import Rules

class TicTacToeRules(Rules):
    def __init__(self):
        pass

    def step(self, board, action, player):
        assert self.get_valid_actions(board, player)[action]
        r = int(action/3)
        c = action % 3
        new_board = board.copy()
        new_board[r,c] = player
        return new_board, -player

    def get_action_space(self):
        return 9

    def get_valid_actions(self, board, player):
        valid_actions = [0] * self.get_action_space()
        for r in range(3):
            for c in range(3):
                if not board[r,c]:
                    valid_actions[r*3 + c] = 1
        return valid_actions

    def get_start_board(self):
        return np.zeros((3, 3))

    def perspective(self, board, player):
        return board * player

    def tostring(self, board):
        return board.tostring()

    def terminal(self, board):
        return sum(self.get_valid_actions(board, 1)) == 0 or sum(self.get_valid_actions(board, -1)) == 0 or self.is_winner(board, 1) or self.is_winner(board, -1)

    def result(self, board, player):
        if self.is_winner(board, player):
            return 1.0
        elif self.is_winner(board, -player):
            return -1.0
        else:
            return 0.0

    def is_winner(self, board, player):
        for i in range(3):
            if board[i,0] == board[i,1] == board[i,2] == player:
                return True
            if board[0,i] == board[1,i] == board[2,i] == player:
                return True
        
        if board[0,0] == board[1,1] == board[2,2] == player:
            return True
        if board[2,0] == board[1,1] == board[0,2] == player:
            return True

        return False

    def name(self):
        return "TicTacToe"
