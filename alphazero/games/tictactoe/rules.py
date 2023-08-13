import numpy as np
from typing import Union

from ...rules import Rules

PLAYER_TO_INDEX = {1: 0, -1: 1}

class TicTacToeRules(Rules):
    def __init__(self):
        super().__init__()

    def get_start_state(self) -> np.ndarray:
        return np.zeros((2, 3, 3), dtype=np.float32)

    def get_action_space(self) -> int:
        return 9

    def get_state_shape(self) -> tuple[int, ...]:
        return (2, 3, 3)

    def get_valid_actions(self, state: np.ndarray, player: int) -> np.ndarray:
        valid_actions = np.zeros(9, dtype=np.float32)
        for i in range(9):
            valid_actions[i] = (state.ravel()[i] == 0 and state.ravel()[i + 9] == 0)
        return valid_actions

    def step(self, state: np.ndarray, action: int, player: int) -> np.ndarray:
        r = int(action / 3)
        c = action % 3
        next_state = state.copy()
        next_state[PLAYER_TO_INDEX[player], r, c] = 1
        return next_state

    def flip_view(self, state: np.ndarray) -> np.ndarray:
        flipped = np.zeros((2, 3, 3), dtype=np.float32)
        flipped[0] = state[1]
        flipped[1] = state[0]
        return flipped

    def hash(self, state: np.ndarray) -> int:
        return hash(state.tostring())

    def get_winner(self, state: np.ndarray) -> Union[int, None]:
        # check if player 1 has won
        if state[0, 0, 0] == state[0, 0, 1] == state[0, 0, 2] == 1:
            return 1
        if state[0, 1, 0] == state[0, 1, 1] == state[0, 1, 2] == 1:
            return 1
        if state[0, 2, 0] == state[0, 2, 1] == state[0, 2, 2] == 1:
            return 1
        if state[0, 0, 0] == state[0, 1, 0] == state[0, 2, 0] == 1:
            return 1
        if state[0, 0, 1] == state[0, 1, 1] == state[0, 2, 1] == 1:
            return 1
        if state[0, 0, 2] == state[0, 1, 2] == state[0, 2, 2] == 1:
            return 1
        if state[0, 0, 0] == state[0, 1, 1] == state[0, 2, 2] == 1:
            return 1
        if state[0, 0, 2] == state[0, 1, 1] == state[0, 2, 0] == 1:
            return 1

        # check if player -1 has won
        if state[1, 0, 0] == state[1, 0, 1] == state[1, 0, 2] == 1:
            return -1
        if state[1, 1, 0] == state[1, 1, 1] == state[1, 1, 2] == 1:
            return -1
        if state[1, 2, 0] == state[1, 2, 1] == state[1, 2, 2] == 1:
            return -1
        if state[1, 0, 0] == state[1, 1, 0] == state[1, 2, 0] == 1:
            return -1
        if state[1, 0, 1] == state[1, 1, 1] == state[1, 2, 1] == 1:
            return -1
        if state[1, 0, 2] == state[1, 1, 2] == state[1, 2, 2] == 1:
            return -1
        if state[1, 0, 0] == state[1, 1, 1] == state[1, 2, 2] == 1:
            return -1
        if state[1, 0, 2] == state[1, 1, 1] == state[1, 2, 0] == 1:
            return -1

        # check if tie
        if np.sum(state) == 9:
            return 0

        # no winner
        return None

    def __str__(self) -> str:
        return "TicTacToe"
