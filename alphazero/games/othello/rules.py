import numpy as np
from typing import Union

from ...rules import Rules

PLAYER_TO_INDEX = {1: 0, -1: 1}
INDEX_TO_PLAYER = {0: 1, 1: -1}

def _get_valids(state: np.ndarray, player: int, r: int, c: int) -> list[tuple[int, int]]:
    moves = []
    for dr, dc in [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]:
        cr = r + dr
        cc = c + dc
        dist = 0

        while cr >= 0 and cr < 8 and cc >= 0 and cc < 8 and state[PLAYER_TO_INDEX[-player], cr, cc] == 1:
            cr += dr
            cc += dc
            dist += 1

        if cr < 0 or cr >= 8 or cc < 0 or cc >= 8:
            continue
        if dist == 0:
            continue

        if state[PLAYER_TO_INDEX[player], cr, cc] == 0 and state[PLAYER_TO_INDEX[-player], cr, cc] == 0:
            moves.append((cr, cc))

    return moves

def _apply_flips(state: np.ndarray, player: int, r: int, c: int) -> None:
    for dr, dc in [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]:
        cr = r + dr
        cc = c + dc
        flips = []

        while cr >= 0 and cr < 8 and cc >= 0 and cc < 8 and state[PLAYER_TO_INDEX[-player], cr, cc] == 1:
            flips.append((cr, cc))
            cr += dr
            cc += dc

        if cr < 0 or cr >= 8 or cc < 0 or cc >= 8:
            continue
        if not (state[PLAYER_TO_INDEX[player], cr, cc] == 1 and state[PLAYER_TO_INDEX[-player], cr, cc] == 0):
            continue
        if len(flips) == 0:
            continue

        for _r, _c in flips:
            state[PLAYER_TO_INDEX[-player], _r, _c] = 0
            state[PLAYER_TO_INDEX[player], _r, _c] = 1


class OthelloRules(Rules):
    def __init__(self):
        super().__init__()

    def get_start_state(self) -> np.ndarray:
        state = np.zeros((2, 8, 8), dtype=np.float32)
        state[0, 3, 3] = 1
        state[0, 4, 4] = 1
        state[1, 3, 4] = 1
        state[1, 4, 3] = 1
        return state

    def get_action_space(self) -> int:
        return 64

    def get_valid_actions(self, state: np.ndarray, player: int) -> np.ndarray:
        valid_actions = np.zeros(64, dtype=np.float32)
        valids = set()

        for a in range(self.get_action_space()):
            r = int(a / 8)
            c = a % 8
            if state[PLAYER_TO_INDEX[player], r, c] == 1:
                moves = _get_valids(state, player, r, c)
                valids.update([r*8 + c for r, c in moves])

        for action in valids:
            valid_actions[action] = 1

        return valid_actions

    def step(self, state: np.ndarray, action: int, player: int) -> np.ndarray:
        valid_actions = self.get_valid_actions(state, player)
        if np.sum(valid_actions) == 0:
            return state.copy()

        r = int(action / 8)
        c = action % 8
        next_state = state.copy()
        next_state[PLAYER_TO_INDEX[player], r, c] = 1
        _apply_flips(next_state, player, r, c)
        return next_state

    def flip_view(self, state: np.ndarray) -> np.ndarray:
        flipped = np.zeros((2, 8, 8), dtype=np.float32)
        flipped[0] = state[1]
        flipped[1] = state[0]
        return flipped

    def hash(self, state: np.ndarray) -> int:
        return hash(state.tostring())

    def get_winner(self, state: np.ndarray) -> Union[int, None]:
        if np.sum(self.get_valid_actions(state, 1)) == 0 and np.sum(self.get_valid_actions(state, -1)) == 0:
            p1 = np.sum(state[0])
            p2 = np.sum(state[1])
            if p1 > p2:
                return 1
            elif p2 > p1:
                return -1
            return 0
        return None

    def __str__(self) -> str:
        return "Othello"
