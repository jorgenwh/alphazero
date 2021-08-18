import numpy as np
from typing import List

from alphazero.rules import Rules

class TicTacToeRules(Rules):
    def __init__(self):
        pass

    def step(self, board: np.ndarray, action: int, player: int) -> np.ndarray:
        assert self.get_valid_actions(board, player)[action]
        r = int(action/3)
        c = action % 3
        next_board = board.copy()
        next_board[r,c] = player
        return next_board

    def get_action_space(self) -> int:
        return 9

    def get_valid_actions(self, board: np.ndarray, player: int) -> List[int]:
        valid_actions = [0] * self.get_action_space()
        for r in range(3):
            for c in range(3):
                if not board[r,c]:
                    valid_actions[r*3 + c] = 1
        return valid_actions

    def get_start_board(self) -> np.ndarray:
        return np.zeros((3, 3))

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

    def __str__(self) -> str:
        return "TicTacToe"
