import numpy as np

from rules import Rules

class Othello_Rules(Rules):
    def __init__(self):
        pass

    def step(self, board, action, player):
        assert self.get_valid_actions(board, player)[action]
        r = int(action / 8)
        c = action % 8
        new_board = board.copy()
        new_board[r,c] = player
        self.perform_flips(new_board, player, r, c)
        return new_board, -player

    def perform_flips(self, board, player, r, c):
        for direction in [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]:
            row = []
            cur_r, cur_c = r + direction[0], c + direction[1]
            while cur_r >= 0 and cur_r < 8 and cur_c >= 0 and cur_c < 8:
                if board[cur_r,cur_c] == -player:
                    row.append((cur_r, cur_c))
                if board[cur_r,cur_c] == 0:
                    break
                if board[cur_r,cur_c] == player:
                    for pos in row:
                        board[pos[0],pos[1]] = player
                    break
                
    def get_action_space(self):
        return 8 ** 2

    def get_valid_actions(self, board, player):
        valid_actions = [0] * self.get_action_space()
        valids = set()

        for r in range(8):
            for c in range(8):
                if board[r,c] == player:
                    moves = self.get_valids(board, player, r, c)
                    self.valids.update(moves)

        for a in valids:
            valid_actions[a] = 1

        return valid_actions

    def get_valids(self, board, player, r, c):
        moves = []
        for direction in [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]:
            cur_r, cur_c = r + direction[0], c + direction[1]
            while cur_r >= 0 and cur_r < 8 and cur_c >= 0 and cur_c < 8:
                if board[cur_r,cur_c] == -player:
                    break
                if board[cur_r,cur_c] == 0:
                    moves.append((cur_r, cur_c))
                if board[cur_r,cur_c] == player:
                    break

        return moves

    def start_board(self):
        board = np.zeros((8, 8))
        board[3,3] = board[4,4] = -1
        board[3,4] = board[4,3] = 1
        return board

    def perspective(self, board, player):
        return board * player

    def get_equal_positions(self, board, pi):
        positions = [(board, pi)]
        # TODO
        return positions
        
    def tostring(self, board):
        return board.tostring()

    def terminal(self, board):
        return sum(self.get_valid_actions(board, 1)) == 0 or sum(self.get_valid_actions(board, -1)) == 0 or self.is_winner(board, 1) or self.is_winner(board, -1)

    def result(self, board, player):
        if self.is_winner(board, player):
            return 1
        elif self.is_winner(board, -player):
            return -1
        else:
            return 0

    def is_winner(self, board, player):
        score = np.sum(board)
        if player == 1:
            return score > 0
        else:
            return score < 0