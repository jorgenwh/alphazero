from PyQt5 import QtGui, QtCore, QtWidgets
import numpy as np
from stockfish import Stockfish

class ChessWindow(QtWidgets.QMainWindow):
    def __init__(self, game_rules, policy, args):
        super().__init__()
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), QtGui.QColor(40, 40, 40))
        self.setPalette(p)

        self.game_rules = game_rules
        self.policy = policy
        self.args = args
        self.stockfish = Stockfish("../chess/stockfish_12_win_x64_bmi2/stockfish_20090216_x64_bmi2")

        self.board = self.game_rules.start_board()
        self.cur_player = 1
        self.size = 8

        self.selected = None

        self.init_window()
        self.fps = 60
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.step)
        self.timer.start(1000 / self.fps)
        self.show()

    def init_window(self):
        self.setFixedSize(self.size*80 + 75, self.size*80 + 75)
        self.centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.centralWidget)
        self.setGeometry(1500, 330, 700, 700)
        self.chess_widget = ChessWidget(self.centralWidget, self)
        self.chess_widget.setGeometry(40, 40, 80*self.size, 80*self.size)

    def step(self):
        self.chess_widget.draw()

    def player_step(self, position):
        if self.selected is None:
            self.selected = position
        else:
            pass
            #valid_actions = self.game_rules.get_valid_actions(self.board, self.cur_player)
            #self.game_rules.step()


class ChessWidget(QtWidgets.QWidget):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.parent = parent
        self.white_pieces = {
            0: "../chess/images/wPawn.png",
            1: "../chess/images/wKnight.png",
            2: "../chess/images/wBishop.png",
            3: "../chess/images/wRook.png",
            4: "../chess/images/wQueen.png",
            5: "../chess/images/wKing.png"
        }
        self.black_pieces = {
            0: "../chess/images/bPawn.png",
            1: "../chess/images/bKnight.png",
            2: "../chess/images/bBishop.png",
            3: "../chess/images/bRook.png",
            4: "../chess/images/bQueen.png",
            5: "../chess/images/bKing.png"
        }
        self.show()

    def draw(self):
        self.repaint()

    def paintEvent(self, event):
        painter = QtGui.QPainter()
        painter.begin(self)
        self.draw_board(painter)
        self.draw_pieces()
        painter.end()

    def draw_pieces(self):
        for r in range(self.app.size):
            for c in range(self.app.size):

                piece, player = None, None
                for i in range(6):
                    if self.app.board[r,c,i]:
                        piece = i
                        player = self.app.board[r,c,i]
                        break

                if player == 1:
                    self.draw_piece(self.white_pieces[piece], c, r)
                elif player == -1:
                    self.draw_piece(self.black_pieces[piece], c, r)

    def draw_piece(self, piece, r, c):
        label = QtWidgets.QLabel(self)
        label.setScaledContents(True)
        pixmap = QtGui.QPixmap(piece)
        label.setPixmap(pixmap)
        gap = self.frameGeometry().width() / self.app.size
        label.setGeometry(r * gap, c * gap, 75, 75)
        label.show()

    def draw_board(self, painter):
        painter.setRenderHints(QtGui.QPainter.Antialiasing)
        painter.setRenderHints(QtGui.QPainter.HighQualityAntialiasing)
        painter.setRenderHint(QtGui.QPainter.TextAntialiasing)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        painter.setPen(QtGui.QPen(QtCore.Qt.black))

        right = self.frameGeometry().width()
        bottom = self.frameGeometry().height()
        gap = right / self.app.size

        light = True
        for r in range(self.app.size):
            light = not light
            for c in range(self.app.size):
                if light:
                    painter.setBrush(QtGui.QBrush(QtGui.QColor(75, 75, 75)))
                else:
                    painter.setBrush(QtGui.QBrush(QtGui.QColor(225, 225, 225)))
                light = not light
                painter.drawRect(r*gap, c*gap, r*gap + gap, c*gap + gap)

        for i in range(self.app.size + 1):
            painter.drawLine(i*gap, 0, i*gap, right)
            painter.drawLine(0, i*gap, bottom, i*gap)

    def mousePressEvent(self, event):
        x = int(event.x() / 80)
        y = int(event.y() / 80)
        position = y*self.app.size + x
        self.app.player_step(position)
        