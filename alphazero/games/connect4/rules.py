import numpy as np
from typing import Union

from ...rules import Rules

PLAYER_TO_INDEX = {1: 0, -1: 1}
INDEX_TO_PLAYER = {0: 1, 1: -1}

class Connect4Rules(Rules):
    def __init__(self):
        super().__init__()

    def get_start_state(self) -> np.ndarray:
        return np.zeros((2, 6, 7), dtype=np.float32)

    def get_action_space(self) -> int:
        return 7

    def get_state_shape(self) -> tuple[int, ...]:
        return (2, 6, 7)

    def get_valid_actions(self, state: np.ndarray, player: int) -> np.ndarray:
        valid_actions = np.zeros(7, dtype=np.float32)
        for action in range(7):
            valid_actions[action] = state[0, 0, action] == 0 and state[1, 0, action] == 0
        return valid_actions

    def step(self, state: np.ndarray, action: int, player: int) -> np.ndarray:
        row = None
        for r in range(5, -1, -1):
            if not state[0, r, action] and not state[1, r, action]:
                row = r
                break
        next_state = state.copy()
        next_state[PLAYER_TO_INDEX[player], row, action] = 1
        return next_state

    def flip_view(self, state: np.ndarray) -> np.ndarray:
        flipped = np.zeros((2, 6, 7), dtype=np.float32)
        flipped[0] = state[1]
        flipped[1] = state[0]
        return flipped

    def hash(self, state: np.ndarray) -> int:
        return hash(state.tostring())

    def get_winner(self, state: np.ndarray) -> Union[int, None]:
        for p in range(2):
            for c in range(7):
                for r in range(6):
                    if c < 4:
                        if state[p,r,c] == state[p,r,c+1] == state[p,r,c+2] == state[p,r,c+3] == 1:
                            return INDEX_TO_PLAYER[p]
                    if r < 3:
                        if state[p,r,c] == state[p,r+1,c] == state[p,r+2,c] == state[p,r+3,c] == 1:
                            return INDEX_TO_PLAYER[p]
                    if c < 4 and r < 3:
                        if state[p,r,c] == state[p,r+1,c+1] == state[p,r+2,c+2] == state[p,r+3,c+3] == 1:
                            return INDEX_TO_PLAYER[p]
                    if c < 4 and r >= 3:
                        if state[p,r,c] == state[p,r-1,c+1] == state[p,r-2,c+2] == state[p,r-3,c+3] == 1:
                            return INDEX_TO_PLAYER[p]
        if np.sum(state) == 42:
            return 0
        return None

    def __str__(self) -> str:
        return "Connect4"
