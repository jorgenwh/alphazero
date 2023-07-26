import numpy as np
from typing import Union

from ...rules import Rules

def _can_move_piece(
        board: np.ndarray, roll: int, piece: int
    ) -> bool:
    piece_idx = np.argmax(board[piece])

    if piece_idx == 0 and roll != 6:
        return False

    # TODO
    if piece_idx + roll > 56:
        pass


class LudoRules(Rules):
    def __init__(self):
        super().__init__()

    def get_start_state(self) -> tuple[np.ndarray, np.ndarray]:
        board = np.zeros((8, 57), dtype=np.float32)
        board[:,0] = 1
        roll = np.zeros(6, dtype=np.float32)
        roll[np.random.randint(low=0, high=6)] = 1
        state = (board, roll)
        return state

    def get_action_space(self) -> int:
        return 4

    def get_valid_actions(self, state: np.ndarray, player: int) -> np.ndarray:
        valid_actions = np.zeros(4, dtype=np.float32)
        board, roll = state
        player_board = board[:4] if player == 1 else board[4:]
        roll = np.argmax(roll[0]) + 1
        for piece in range(4):
            if _can_move_piece(player_board, roll, piece):
                valid_actions[piece] = 1
        return valid_actions

    def step(self, 
            state: tuple[np.ndarray, np.ndarray], action: int, player: int
        ) -> tuple[np.ndarray, np.ndarray]:
        pass

    def flip_view(self, 
            state: tuple[np.ndarray, np.ndarray]
        ) -> tuple[np.ndarray, np.ndarray]:
        flipped = np.zeros((8, 57), dtype=np.float32)
        flipped[:4] = state[4:]
        flipped[4:] = state[:4]
        return flipped

    def hash(self, state: tuple[np.ndarray, np.ndarray]) -> int:
        return hash(state[0].tostring() + state[1].tostring())

    def get_winner(self, state: tuple[np.ndarray, np.ndarray]) -> Union[int, None]:
        pass

    def __str__(self) -> str:
        return "Ludo"
