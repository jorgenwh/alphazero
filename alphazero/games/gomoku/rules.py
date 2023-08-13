import numpy as np
from typing import Union

from ...rules import Rules

PLAYER_TO_INDEX = {1: 0, -1: 1}
INDEX_TO_PLAYER = {0: 1, 1: -1}

GOMOKU_BOARD_SIZE = 9

class GomokuRules(Rules):
    def __init__(self):
        super().__init__()

    def get_start_state(self) -> np.ndarray:
        return np.zeros((2, GOMOKU_BOARD_SIZE, GOMOKU_BOARD_SIZE), dtype=np.float32)

    def get_action_space(self) -> int:
        return GOMOKU_BOARD_SIZE**2

    def get_state_shape(self) -> tuple[int, ...]:
        return (2, GOMOKU_BOARD_SIZE, GOMOKU_BOARD_SIZE)

    def get_valid_actions(self, state: np.ndarray, player: int) -> np.ndarray:
        valid_actions = np.zeros(GOMOKU_BOARD_SIZE**2, dtype=np.float32)
        for r in range(GOMOKU_BOARD_SIZE):
            for c in range(GOMOKU_BOARD_SIZE):
                if state[0,r,c] == 0 and state[1,r,c] == 0:
                    valid_actions[r*GOMOKU_BOARD_SIZE + c] = 1
        return valid_actions

    def step(self, state: np.ndarray, action: int, player: int) -> np.ndarray:
        r = action // GOMOKU_BOARD_SIZE
        c = action % GOMOKU_BOARD_SIZE
        next_state = state.copy()
        next_state[PLAYER_TO_INDEX[player], r, c] = 1
        return next_state

    def flip_view(self, state: np.ndarray) -> np.ndarray:
        flipped = np.zeros((2, GOMOKU_BOARD_SIZE, GOMOKU_BOARD_SIZE), dtype=np.float32)
        flipped[0] = state[1]
        flipped[1] = state[0]
        return flipped

    def hash(self, state: np.ndarray) -> int:
        return hash(state.tostring())

    def get_winner(self, state: np.ndarray) -> Union[int, None]:
        for c in range(GOMOKU_BOARD_SIZE):
            for r in range(GOMOKU_BOARD_SIZE):
                if c <= GOMOKU_BOARD_SIZE - 5:
                    if np.sum(state[0,r,c:c+5]) == 5:
                        return 1
                    if np.sum(state[1,r,c:c+5]) == 5:
                        return -1
                if r <= GOMOKU_BOARD_SIZE - 5:
                    if np.sum(state[0,r:r+5,c]) == 5:
                        return 1
                    if np.sum(state[1,r:r+5,c]) == 5:
                        return -1
                if c <= GOMOKU_BOARD_SIZE - 5 and r <= GOMOKU_BOARD_SIZE - 5:
                    if state[0,r,c] == state[0,r+1,c+1] == state[0,r+2,c+2] == state[0,r+3,c+3] == state[0,r+4,c+4] == 1:
                        return 1
                    if state[1,r,c] == state[1,r+1,c+1] == state[1,r+2,c+2] == state[1,r+3,c+3] == state[1,r+4,c+4] == 1:
                        return -1
                if c <= GOMOKU_BOARD_SIZE - 5 and r >= 4:
                    if state[0,r,c] == state[0,r-1,c+1] == state[0,r-2,c+2] == state[0,r-3,c+3] == state[0,r-4,c+4] == 1:
                        return 1
                    if state[1,r,c] == state[1,r-1,c+1] == state[1,r-2,c+2] == state[1,r-3,c+3] == state[1,r-4,c+4] == 1:
                        return -1
        if np.sum(state) == GOMOKU_BOARD_SIZE**2:
            return 0
        return None

    def __str__(self) -> str:
        return "Gomoku"
