import numpy as np
from typing import Union

class Rules:
    def __init__(self):
        pass

    def get_start_state(self) -> np.ndarray:
        raise NotImplementedError

    def get_action_space(self) -> int:
        raise NotImplementedError

    def get_state_shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    def get_valid_actions(self, state: np.ndarray, player: int) -> np.ndarray:
        raise NotImplementedError

    def step(self, state: np.ndarray, action: int, player: int) -> np.ndarray:
        raise NotImplementedError

    def flip_view(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def hash(self, state: np.ndarray) -> int:
        raise NotImplementedError

    def get_winner(self, state: np.ndarray) -> Union[int, None]:
        raise NotImplementedError

    def __str__(self) -> str:
        raise NotImplementedError
