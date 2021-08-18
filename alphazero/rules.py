import numpy as np
from typing import List

class Rules():
    """
    Abstract game-rules class.
    For any new game, inherit from this class and implement all of the below methods.
    """
    def __init__(self):
        pass

    def step(self, board: np.ndarray, action: int, player: int) -> np.ndarray:
        raise NotImplementedError

    def get_action_space(self) -> int:
        raise NotImplementedError

    def get_valid_actions(self, board: np.ndarray, player: int) -> List[int]:
        raise NotImplementedError

    def get_start_board(self) -> np.ndarray:
        raise NotImplementedError

    def flip(self, board: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def to_string(self, board: np.ndarray) -> str:
        raise NotImplementedError

    def is_concluded(self, board: np.ndarray) -> bool:
        raise NotImplementedError

    def get_result(self, board: np.ndarray) -> float:
        raise NotImplementedError

    def has_won(self, board: np.ndarray, player: int) -> bool:
        raise NotImplementedError

    def __str__(self) -> str:
        raise NotImplementedError
