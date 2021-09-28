import numpy as np
from typing import List

from alphazero.rules import Rules

class GomokuRules(Rules):
    def __init__(self, size: int):
        self.size = size

    def step(self, board: np.ndarray, action: int, player: int) -> np.ndarray:
        assert self.get_valid_actions(board, player)[action]
        r = int(action / self.size)
        c = action % self.size
        next_board = board.copy()
        next_board[r,c] = player
        return next_board

    def get_action_space(self) -> int:
        return self.size ** 2

    def get_valid_actions(self, board: np.ndarray, player: int) -> List[int]:
        valid_actions = [0] * self.get_action_space()
        for r in range(self.size):
            for c in range(self.size):
                if not board[r,c]:
                    valid_actions[r*self.size + c] = 1
        return valid_actions

    def get_start_board(self) -> np.ndarray:
        return np.zeros((self.size, self.size))

    def flip(self, board: np.ndarray) -> np.ndarray:
        return board * -1 

    def to_string(self, board: np.ndarray) -> str:
        return board.tostring()

    def is_concluded(self, board: np.ndarray) -> bool:
        return sum(self.get_valid_actions(board, 1)) == 0 or sum(self.get_valid_actions(board, -1)) == 0 or self.has_won(board, 1) or self.has_won(board, -1)

    def get_result(self, board: np.ndarray) -> float:
        if self.has_won(board, 1):
            return 1
        elif self.has_won(board, -1):
            return -1
        else:
            return 0

    def has_won(self, board: np.ndarray, player: int) -> bool:
        for c in range(self.size - 4):
            for r in range(self.size):
                if board[r,c] == board[r,c+1] == board[r,c+2] == board[r,c+3] == board[r,c+4] == player:
                    return True

        for c in range(self.size):
            for r in range(self.size - 4):
                if board[r,c] == board[r+1,c] == board[r+2,c] == board[r+3,c] == board[r+4,c] == player:
                    return True

        for c in range(self.size - 4):
            for r in range(self.size - 4):
                if board[r,c] == board[r+1,c+1] == board[r+2,c+2] == board[r+3,c+3] == board[r+4,c+4] == player:
                    return True

        for c in range(self.size - 4):
            for r in range(4, self.size):
                if board[r,c] == board[r-1,c+1] == board[r-2,c+2] == board[r-3,c+3] == board[r-4,c+4] == player:
                    return True

        return False

    def __str__(self) -> str:
        return "Gomoku"