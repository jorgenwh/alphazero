import numpy as np
from typing import List

from alphazero.rules import Rules

class Connect4Rules(Rules):
    def __init__(self):
        pass

    def step(self, board: np.ndarray, action: int, player: int) -> np.ndarray:
        assert self.get_valid_actions(board, player)[action]
        row = self.get_action_row(board, action)
        next_board = board.copy()
        next_board[row,action] = player
        return next_board

    def get_action_space(self) -> int:
        return 7

    def get_action_row(self, board: np.ndarray, action: int) -> int:
        for row in range(5, -1, -1):
            if not board[row,action]:
                return row

    def get_valid_actions(self, board: np.ndarray, player: int) -> List[int]:
        valid_actions = [0] * 7
        for action in range(7):
            if not board[0,action]:
                valid_actions[action] = 1
        return valid_actions

    def get_start_board(self) -> np.ndarray:
        return np.zeros((6, 7))

    def flip(self, board: np.ndarray) -> np.ndarray:
        return board * -1

    def to_string(self, board: np.ndarray) -> str:
        return board.tostring()

    def is_concluded(self, board: np.ndarray) -> bool:
        return sum(self.get_valid_actions(board, 1)) == 0 or \
                sum(self.get_valid_actions(board, -1)) == 0 or \
                self.has_won(board, 1) or \
                self.has_won(board, -1)

    def get_result(self, board: np.ndarray) -> float:
        if self.has_won(board, 1):
            return 1
        elif self.has_won(board, -1):
            return -1
        else:
            return 0

    def has_won(self, board: np.ndarray, player: int) -> bool:
        for c in range(7):
            for r in range(6):
                if c < 4:
                    if board[r,c] == board[r,c+1] == board[r,c+2] == board[r,c+3] == player:
                        return True
                if r < 3:
                    if board[r,c] == board[r+1,c] == board[r+2,c] == board[r+3,c] == player:
                        return True
                if c < 4 and r < 3:
                    if board[r,c] == board[r+1,c+1] == board[r+2,c+2] == board[r+3,c+3] == player:
                        return True
                if c < 4 and r >= 3:
                    if board[r,c] == board[r-1,c+1] == board[r-2,c+2] == board[r-3,c+3] == player:
                        return True
        return False

    def __str__(self) -> str:
        return "Connect 4"
