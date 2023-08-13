import numpy as np
from typing import Union

from ...rules import Rules

PIECE_MASK = {0: [1, 2, 3], 1: [0, 2, 3], 2: [0, 1, 3], 3: [0, 1, 2]}

RED_TO_YELLOW_CAPTURE_INDICES = [
    None, 
    27, 28, 29, 30, 31, 
    32, 33, 34, 35, 36, 37,
    38, 
    39, 40, 41, 42, 43, 44,
    45, 46, 47, 48, 49, 50,
    51, 
    None, None, 2, 3, 4, 5,
    6, 7, 8, 9, 10, 11,
    12, 
    13, 14, 15, 16, 17, 18,
    19, 20, 21, 22, 23, 24,
    25, 
    None, None, None, None, None
]


def _can_move_piece(
        board: np.ndarray, roll: int, piece: int
    ) -> bool:
    if np.sum(board[piece]) == 0:
        return False

    piece_idx = int(np.argmax(board[piece]))

    if piece_idx == 0 and roll != 6:
        return False

    land_idx = piece_idx + roll
    if land_idx > 57:
        land_idx = 57 - (land_idx - 57)
    if piece_idx == 0:
        land_idx = 1

    other_pieces = PIECE_MASK[piece]
    for p in other_pieces:
        if np.argmax(board[p]) == land_idx:
            return False

    return True


class LudoRules(Rules):
    def __init__(self):
        super().__init__()

    def get_start_state(self) -> tuple[np.ndarray, np.ndarray]:
        board = np.zeros((8, 57), dtype=np.float32)
        board[:,0] = 1
        roll = np.zeros(6, dtype=np.float32)
        roll_idx = np.random.randint(low=0, high=6)
        roll[roll_idx] = 1
        state = (board, roll)
        return state

    def get_action_space(self) -> int:
        return 4

    def get_state_shape(self) -> tuple[int, ...]:
        return (8*57 + 6,)

    def get_valid_actions(self, state: np.ndarray, player: int) -> np.ndarray:
        valid_actions = np.zeros(4, dtype=np.float32)
        board, roll = state
        player_board = board[:4] if player == 1 else board[4:]
        roll = int(np.argmax(roll)) + 1
        for piece in range(4):
            if _can_move_piece(player_board, roll, piece):
                valid_actions[piece] = 1
        return valid_actions

    def step(self, 
            state: tuple[np.ndarray, np.ndarray], action: int, player: int
        ) -> tuple[np.ndarray, np.ndarray]:
        valid_actions = self.get_valid_actions(state, player)

        board, roll = state
        next_board = board.copy()
        next_roll = np.zeros(6, dtype=np.float32)
        next_roll[np.random.randint(low=0, high=6)] = 1

        if np.sum(valid_actions) == 0:
            next_state = (next_board, next_roll)
            return next_state

        piece = action if player == 1 else action + 4
        piece_idx = int(np.argmax(next_board[piece]))

        land_idx = piece_idx + int(np.argmax(roll)) + 1
        if land_idx > 57:
            land_idx = 57 - (land_idx - 57)
        if piece_idx == 0:
            land_idx = 1

        if land_idx == 57:
            next_board[piece, piece_idx] = 0
            next_state = (next_board, next_roll)
            return next_state

        next_board[piece, piece_idx] = 0
        next_board[piece, land_idx] = 1

        if player == 1:
            capture_idx = RED_TO_YELLOW_CAPTURE_INDICES[land_idx]
        else:
            if land_idx == 1:
                capture_idx = 27
            elif land_idx == 26 or land_idx == 27 or land_idx > 51:
                capture_idx = None
            elif land_idx in RED_TO_YELLOW_CAPTURE_INDICES:
                capture_idx = RED_TO_YELLOW_CAPTURE_INDICES.index(land_idx)
            else:
                capture_idx = None
        if capture_idx is not None:
            if player == 1:
                for i in range(4, 8):
                    if np.argmax(next_board[i]) == capture_idx:
                        next_board[i, capture_idx] = 0
                        next_board[i, 0] = 1
            else:
                for i in range(4):
                    if np.argmax(next_board[i]) == capture_idx:
                        next_board[i, capture_idx] = 0
                        next_board[i, 0] = 1

        next_state = (next_board, next_roll)
        return next_state

    def flip_view(self, 
            state: tuple[np.ndarray, np.ndarray]
        ) -> tuple[np.ndarray, np.ndarray]:
        board, roll = state
        flipped = np.zeros((8, 57), dtype=np.float32)
        flipped[:4] = board[4:]
        flipped[4:] = board[:4]
        flipped_state = (flipped, roll)
        return flipped_state

    def hash(self, state: tuple[np.ndarray, np.ndarray]) -> int:
        return hash(state[0].tostring() + state[1].tostring())

    def get_winner(self, state: tuple[np.ndarray, np.ndarray]) -> Union[int, None]:
        board = state[0]
        if np.sum(board[:4]) == 0:
            return 1
        if np.sum(board[4:]) == 0:
            return -1
        return None

    def __str__(self) -> str:
        return "Ludo"
