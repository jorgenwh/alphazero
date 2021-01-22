import re
import chess
import numpy as np
from rules import Rules

class ChessRules(Rules):
    def __init__(self):
        self.n_to_piece = {0:"p", 1:"n", 2:"b", 3:"r", 4:"q", 5:"k"}
        self.piece_to_n = {"p":0, "n":1, "b":2, "r":3, "q":4, "k":5}
        self.files = {"a":0, "b":1, "c":2, "d":3, "e":4, "f":5, "g":6, "h":7}
        self.n_to_file = {0:"a", 1:"b", 2:"c", 3:"d", 4:"e", 5:"f", 6:"g", 7:"h"}

    def step(self, board, action, player):
        assert board[6,0,0] == player
        assert self.get_valid_actions(board, player)[action]

        fen = self.numpy_to_fen(board)
        cb = chess.Board(fen)
        uci = self.action_to_uci(action)
        move = chess.Move.from_uci(uci)
        cb.push(move)
        next_fen = cb.fen()
        b = self.fen_to_numpy(next_fen)
        return b, -player

    def get_action_space(self):
        return 4096

    def get_valid_actions(self, board, player):
        valid_actions = np.zeros(self.get_action_space())
        fen = self.numpy_to_fen(board)
        cb = chess.Board(fen)
        for action in cb.legal_moves:
            valid_actions[self.uci_to_action(action.uci())] = 1
        return valid_actions

    def start_board(self):
        """
        Board representation
        matrix level:
            0: pawns {1, 0, -1}
            1: knights {1, 0, -1}
            2: bishops {1, 0, -1}
            3: rooks {1, 0, -1}
            4: queens {1, 0, -1}
            5: kings {1, 0, -1}
            6: current player {1, -1}
            7: castling - white king side
            8: castling - white queen side
            9: castling - black king side
            10: castling - black queen side
            11: en-passant
        """
        board = np.zeros((12, 8, 8))

        # pawns
        board[0,1,:] = -1
        board[0,6,:] = 1

        # knights
        board[1,0,1] = board[1,0,6] = -1
        board[1,7,1] = board[1,7,6] = 1

        # place bishops
        board[2,0,2] = board[2,0,5] = -1
        board[2,7,2] = board[2,7,5] = 1

        # rooks
        board[3,0,0] = board[3,0,7] = -1
        board[3,7,0] = board[3,7,7] = 1

        # kings and queens
        board[4,0,3] = board[5,0,4] = -1
        board[4,7,3] = board[5,7,4] = 1

        # current turn
        board[6,:,:] = 1

        # castling ability
        board[7:11,:,:] = 1

        return board

    def perspective(self, board, player):
        fen_before = self.numpy_to_fen(board)
        board_before = chess.Board(fen_before)
        print("board before perspective")
        print(board_before)
        print()
        b = board.copy()

        if player == 1:
            return b

        b[:7,:,:] *= -1
        b[7,:,:], b[8,:,:] = b[9,:,:], b[10,:,:]
        b = np.flip(b, 1)
        fen_after = self.numpy_to_fen(b)
        board_after = chess.Board(fen_after)
        print("board after perspective")
        print(board_after)
        print("\n\n")
        return b

    def tostring(self, board):
        return board.tostring()

    def terminal(self, board):
        fen = self.numpy_to_fen(board)
        cb = chess.Board(fen)
        return cb.is_game_over()

    def result(self, board, player):
        if self.is_winner(board, player):
            return 1.0
        elif self.is_winner(board, -player):
            return -1.0
        return 0.0

    def is_winner(self, board, player):
        fen = self.numpy_to_fen(board)
        cb = chess.Board(fen)
        b = chess.Board()
        r = b.result()

        if r == "1-0" and player == 1:
            return True
        elif r == "0-1" and player == -1:
            return True
        return False

    def numpy_to_fen(self, board):
        fen = ""

        for i in range(8):
            row = ""
            empty = 0

            for j in range(8):
                board_pos = board[:,i,j]
                player, piece = None, None

                for k in range(6):
                    if board_pos[k]:
                        player = board_pos[k]
                        piece = k
                        break
                
                if player is None:
                    empty += 1
                else:
                    piece = self.n_to_piece[piece]

                    if player == 1:
                        piece = piece.upper()
                    if empty > 0:
                        row += str(empty)

                    empty = 0
                    row += piece

            if empty > 0:
                row += str(empty)

            fen += row + "/"
        fen = fen[:-1]

        player = "w" if board[6,0,0] == 1 else "b"
        fen += " " + player

        kw = "K" if board[7,0,0] else ""
        qw = "Q" if board[8,0,0] else ""
        kb = "k" if board[9,0,0] else ""
        qb = "q" if board[10,0,0] else ""

        castling = "-" if not (kw or qw or kb or qb) else kw + qw + kb + qb
        fen += " " + castling + " - 0 0"
        return fen

    def fen_to_numpy(self, fen):
        board = np.zeros((12, 8, 8))
        regex = "^(.*?)\ "
        rows = re.findall(regex, fen)[0].split("/")
        
        for i, row in enumerate(rows):
            j = 0
            k = 0

            while k < len(row):
                if row[j].isnumeric():
                    k += int(row[j])
                else:
                    piece = self.piece_to_n[row[j].lower()]
                    player = 1 if row[j].isupper() else -1
                    board[piece,i,k] = player
                    k += 1

        data = fen.split(" ")
        move = data[1]
        castling = data[2]
        en_passant = data[3]

        move = 1 if move == "w" else -1
        kw = int("K" in castling)
        qw = int("Q" in castling)
        kb = int("k" in castling)
        qb = int("q" in castling)

        board[6,:,:] = move
        board[7,:,:] = kw
        board[8,:,:] = qw
        board[9,:,:] = kb
        board[10,:,:] = qb

        if en_passant != "-":
            i = int(en_passant[1])
            j = self.files[en_passant[0]]
            board[11,i,j] = 1

        return board

    def uci_to_action(self, uci):
        sq1 = self.files[uci[0]] * 8 + int(uci[1])
        sq2 = self.files[uci[2]] * 8 + int(uci[3])
        return sq1 * 64 + sq2

    def action_to_uci(self, action):
        sq1 = int(action / 64)
        sq2 = action % 64
        
        p1f = int(sq1 / 8)
        p1r = sq1 % 8
        sq1 = self.n_to_file[p1f] + str(p1r)

        p2f = int(sq2 / 8)
        p2r = sq2 % 8
        sq2 = self.n_to_file[p2f] + str(p2r)

        return sq1 + sq2

    def name(self):
        return "chess"
