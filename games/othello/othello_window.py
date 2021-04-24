from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtGui import QFont
import numpy as np
from mcts import MCTS

class OthelloWindow(QtWidgets.QMainWindow):
    def __init__(self, game_rules, policy, args):
        super().__init__()
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), QtGui.QColor(80, 40, 40))
        self.setPalette(p)

        self.game_rules = game_rules
        self.policy = policy
        self.args = args
        self.cur_player = 1
        self.nnet_turn = -1
        self.board = self.game_rules.get_start_board()
        self.size = self.args.othello_size

        self.init_window()
        self.fps = 200
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.step)
        self.timer.start(1000 / self.fps)
        self.show()

    def init_window(self):
        self.setFixedSize(65*self.size + 50, 65*self.size + 50)
        self.centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.centralWidget)
        self.setGeometry(750, 280, 65*self.size + 50, 65*self.size + 50)

        self.othello_widget = OthelloWidget(self.centralWidget, self)
        self.othello_widget.setGeometry(25, 25, 65*self.size, 65*self.size)

    def step(self):
        if self.cur_player == self.nnet_turn and not self.game_rules.terminal(self.board):
            board_perspective = self.game_rules.perspective(self.board, self.cur_player)
            pi = self.policy.get_policy(board_perspective, t=0)
            action = np.argmax(pi)
            self.board, self.cur_player = self.game_rules.step(self.board, action, self.cur_player)
            self.othello_widget.draw()

        elif not self.game_rules.terminal(self.board):
            if sum(self.game_rules.get_valid_actions(self.board, -self.nnet_turn)) == 0:
                self.board, self.cur_player = self.game_rules.step(self.board, 0, self.cur_player)
                self.othello_widget.draw()

    def player_step(self, action):
        if self.game_rules.terminal(self.board):
            self.board = self.game_rules.get_start_board()
            self.cur_player = 1
            self.nnet_turn *= -1
            self.othello_widget.draw()
        else:
            if self.game_rules.get_valid_actions(self.board, self.cur_player)[action] and not self.game_rules.terminal(self.board):
                self.board, self.cur_player = self.game_rules.step(self.board, action, self.cur_player)
                self.othello_widget.draw()


class OthelloWidget(QtWidgets.QWidget):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), QtGui.QColor(60, 130, 60))
        self.setPalette(p)
        self.app = app
        self.winner = None
        self.show()

    def draw(self):
        if self.app.game_rules.terminal(self.app.board):
            self.winner = self.app.game_rules.result(self.app.board, 1)
            if self.winner == 1:
                winner = "BLACK"
            elif self.winner == -1:
                winner = "WHITE"
            else:
                winner = "NONE"
            b, w = self.get_scores(self.app.board)
            print(f"Winner: {winner} - Score B/W: {b}/{w}")
        else:
            self.winner = None

        self.repaint()

    def get_scores(self, board):
        black = white = 0
        for r in range(self.app.size):
            for c in range(self.app.size):
                if board[r,c] == 1:
                    black += 1
                elif board[r,c] == -1:
                    white += 1

        return black, white

    def paintEvent(self, event):
        painter = QtGui.QPainter()
        painter.begin(self)
        self.draw_board(painter)
        self.draw_stones(painter)
        painter.end()

    def draw_board(self, painter):
        painter.setRenderHints(QtGui.QPainter.Antialiasing)
        painter.setRenderHints(QtGui.QPainter.HighQualityAntialiasing)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        painter.setPen(QtGui.QPen(QtCore.Qt.black))

        right = self.frameGeometry().width()
        bottom = self.frameGeometry().height()
        gap = right / self.app.size

        for i in range(self.app.size + 1):
            painter.drawLine(i*gap, 0, i*gap, right)
            painter.drawLine(0, i*gap, bottom, i*gap)

    def draw_stones(self, painter):
        valid_actions = self.app.game_rules.get_valid_actions(self.app.board, self.app.cur_player)

        for r in range(self.app.size):
            for c in range(self.app.size):
                if self.app.board[r,c] == 1:
                    if self.winner == -1 or self.winner == 0:
                        self.draw_black(painter, (r, c), 0.6)
                    else:
                        self.draw_black(painter, (r, c), 1)
                elif self.app.board[r,c] == -1:
                    if self.winner == 1 or self.winner == 0:
                        self.draw_white(painter, (r, c), 0.6)
                    else:
                        self.draw_white(painter, (r, c), 1)

                if valid_actions[r * self.app.size + c]:
                    if self.app.cur_player == 1:
                        self.draw_black(painter, (r, c), 0.2)
                    elif self.app.cur_player == -1:
                        self.draw_white(painter, (r, c), 0.2)
                
    def draw_black(self, painter, intersection, opacity):
        painter.setBrush(QtGui.QBrush(QtGui.QColor(40, 40, 40)))
        painter.setOpacity(opacity)
        gap = self.frameGeometry().width() / self.app.size
        x = 8.5 + intersection[1] * gap
        y = 8.5 + intersection[0] * gap
        painter.drawEllipse(x, y, gap*0.75, gap*0.75)
    
    def draw_white(self, painter, intersection, opacity):
        painter.setBrush(QtGui.QBrush(QtGui.QColor(215, 215, 215))) 
        painter.setOpacity(opacity)
        gap = self.frameGeometry().width() / self.app.size
        x = 8.5 + intersection[1] * gap
        y = 8.5 + intersection[0] * gap
        painter.drawEllipse(x, y, gap*0.75, gap*0.75)

    def mousePressEvent(self, event):
        x = int(event.x() / 65)
        y = int(event.y() / 65)
        action = y*self.app.size + x
        self.app.player_step(action)