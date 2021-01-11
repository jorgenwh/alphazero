import numpy as np

from rules import Rules

class Othello_Rules(Rules):
    def __init__(self, size):
        self.size = size

    def step(self, board, action, player):
        valid_actions = self.get_valid_actions(board, player)
        if sum(valid_actions) == 0:
            return board.copy(), -player

        assert valid_actions[action]
        r = int(action / self.size)
        c = action % self.size
        new_board = board.copy()
        new_board[r,c] = player
        self.flips(new_board, player, r, c)
        return new_board, -player

    def flips(self, board, player, r, c):
        for direction in [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]:
            cur_r, cur_c = r + direction[0], c + direction[1]
            row = [(r, c)]

            while cur_r >= 0 and cur_r < self.size and cur_c >= 0 and cur_c < self.size:
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
                
    def get_action_space(self):
        return self.size ** 2

    def get_valid_actions(self, board, player):
        valid_actions = [0] * self.get_action_space()
        valids = set()
        
        for a in range(self.get_action_space()):
            r, c = int(a / self.size), a % self.size
            if board[r,c] == player:
                moves = self.get_valids(board, player, r, c)
                valids.update([pos[0] * self.size + pos[1] for pos in moves])

        for a in valids:
            valid_actions[a] = 1

        return valid_actions

    def get_valids(self, board, player, r, c):
        moves = []
        for direction in [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]:
            cur_r, cur_c = r + direction[0], c + direction[1]
            row = [board[r,c]]

            while cur_r >= 0 and cur_r < self.size and cur_c >= 0 and cur_c < self.size:
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

    def start_board(self):
        board = np.zeros((self.size, self.size))
        board[(self.size // 2) - 1,(self.size // 2) - 1] = board[(self.size // 2),(self.size // 2)] = -1
        board[(self.size // 2) - 1,(self.size // 2)] = board[(self.size // 2),(self.size // 2) - 1] = 1
        return board

    def perspective(self, board, player):
        return board * player

    def tostring(self, board):
        return board.tostring()

    def terminal(self, board):
        return sum(self.get_valid_actions(board, 1)) == 0 and sum(self.get_valid_actions(board, -1)) == 0

    def result(self, board, player):
        if self.is_winner(board, player):
            return 1.0
        elif self.is_winner(board, -player):
            return -1.0
        else:
            return 0.0

    def is_winner(self, board, player):
        score = np.sum(board)
        if player == 1:
            return score > 0
        else:
            return score < 0
            
    def name(self):
        return "Othello"
        