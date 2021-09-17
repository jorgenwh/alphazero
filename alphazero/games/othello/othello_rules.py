import numpy as np
from typing import List, Tuple

from alphazero.rules import Rules

class OthelloRules(Rules):
    def __init__(self):
        pass

    def step(self, board: np.ndarray, action: int, player: int) -> np.ndarray:
        valid_actions = self.get_valid_actions(board, player)
        if sum(valid_actions) == 0:
            return board.copy()

        assert valid_actions[action]
        r = int(action / 8)
        c = action % 8
        next_board = board.copy()
        next_board[r,c] = player
        self.flips(next_board, player, r, c)
        return next_board

    def flips(self, board: np.ndarray, player: int, r: int, c: int) -> None:
        for direction in [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]:
            cur_r, cur_c = r + direction[0], c + direction[1]
            row = [(r, c)]

            while cur_r >= 0 and cur_r < 8 and cur_c >= 0 and cur_c < 8:
                row.append((cur_r, cur_c))
                if board[cur_r,cur_c] != -player:
                    break
                cur_r += direction[0]
                cur_c += direction[1]
    
            if len(row) < 3:
                continue
            if board[row[-1][0],row[-1][1]] != player:
                continue
            for r_, c_ in row:
                board[r_,c_] = player
                
    def get_action_space(self) -> int:
        return 64 

    def get_valid_actions(self, board: np.ndarray, player: int) -> List[int]:
        valid_actions = [0] * self.get_action_space()
        valids = set()
        
        for a in range(self.get_action_space()):
            r, c = int(a / 8), a % 8 
            if board[r,c] == player:
                moves = self.get_valids(board, player, r, c)
                valids.update([pos[0] * 8 + pos[1] for pos in moves])

        for a in valids:
            valid_actions[a] = 1

        return valid_actions

    def get_valids(self, board: np.ndarray, player: int, r: int, c: int) -> List[Tuple[int, int]]:
        moves = []
        for direction in [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]:
            cur_r, cur_c = r + direction[0], c + direction[1]
            row = [board[r,c]]

            while cur_r >= 0 and cur_r < 8 and cur_c >= 0 and cur_c < 8:
                row.append(board[cur_r,cur_c])
                if row[-1] != -player:
                    break
                cur_r += direction[0]
                cur_c += direction[1]

            if len(row) < 3:
                continue
            if not (row[0] == player and row[-1] == 0):
                continue
            if player in row[1:-1] or 0 in row[1:-1]:
                continue
            moves.append((cur_r,cur_c))

        return moves

    def get_start_board(self) -> np.ndarray:
        board = np.zeros((8, 8))
        board[(8 // 2) - 1,(8 // 2) - 1] = board[(8 // 2),(8 // 2)] = -1
        board[(8 // 2) - 1,(8 // 2)] = board[(8 // 2),(8 // 2) - 1] = 1
        return board

    def flip(self, board: np.ndarray) -> np.ndarray:
        return board * -1 

    def to_string(self, board: np.ndarray) -> str:
        return board.tostring()

    def is_concluded(self, board: np.ndarray) -> bool:
        return sum(self.get_valid_actions(board, 1)) == 0 and sum(self.get_valid_actions(board, -1)) == 0

    def get_result(self, board: np.ndarray) -> float:
        if self.has_won(board, 1):
            return 1
        elif self.has_won(board, -1):
            return -1
        else:
            return 0

    def has_won(self, board: np.ndarray, player: int) -> bool:
        score = np.sum(board)
        if player == 1:
            return score > 0
        else:
            return score < 0
            
    def __str__(self) -> str:
        return "Othello"
