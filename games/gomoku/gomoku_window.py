from PyQt5 import QtGui, QtCore, QtWidgets
import numpy as np

class Gomoku_Window(QtWidgets.QMainWindow):
    def __init__(self, game_rules, policy, args):
        super().__init__()
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), QtGui.QColor(235, 200, 100))
        self.setPalette(p)

        self.game_rules = game_rules
        self.policy = policy
        self.args = args
        self.size = self.args.size
        self.cur_player = 1
        self.nnet_turn = -1
        self.board = self.game_rules.start_board()

        self.init_window()
        self.fps = 200
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.step)
        self.timer.start(1000 / self.fps)
        self.show()

    def init_window(self):
        self.setFixedSize(self.size*40 + 75, self.size*40 + 75)
        self.centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.centralWidget)
        self.setGeometry(1500, 330, 350, 350)
        self.gomoku_widget = Gomoku_Widget(self.centralWidget, self)
        self.gomoku_widget.setGeometry(40, 40, 40*self.size, 40*self.size)

    def step(self):
        if self.cur_player == self.nnet_turn and not self.game_rules.terminal(self.board):
            board_perspective = self.game_rules.perspective(self.board, self.cur_player)
            pi = self.policy.get_policy(board_perspective, t=0)
            action = np.argmax(pi)
            self.board, self.cur_player = self.game_rules.step(self.board, action, self.cur_player)
            self.gomoku_widget.draw()

    def player_step(self, action):
        if self.game_rules.terminal(self.board):
            self.board = self.game_rules.start_board()
            self.cur_player = 1
            self.nnet_turn *= -1
            self.gomoku_widget.draw()
        else:
            if self.game_rules.get_valid_actions(self.board, self.cur_player)[action] and not self.game_rules.terminal(self.board):
                self.board, self.cur_player = self.game_rules.step(self.board, action, self.cur_player)
                self.gomoku_widget.draw()


class Gomoku_Widget(QtWidgets.QWidget):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.winner_row = []
        self.show()

    def draw(self):
        self.repaint()

    def paintEvent(self, event):
        painter = QtGui.QPainter()
        painter.begin(self)
        self.winner_row = self.get_winner_row()
        self.draw_board(painter)
        self.draw_stones(painter)
        painter.end()

    def draw_board(self, painter):
        painter.setRenderHints(QtGui.QPainter.Antialiasing)
        painter.setRenderHints(QtGui.QPainter.HighQualityAntialiasing)
        painter.setRenderHint(QtGui.QPainter.TextAntialiasing)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        painter.setPen(QtGui.QPen(QtCore.Qt.black))

        right = self.frameGeometry().width()
        bottom = self.frameGeometry().height()
        gap = right / self.app.size

        for i in range(self.app.size):
            painter.drawLine(gap/2 + i*gap, 0, gap/2 + i*gap, right)
            painter.drawLine(0, gap/2 + i*gap, bottom, gap/2 + i*gap)

    def draw_stones(self, painter):
        for r in range(self.app.size):
            for c in range(self.app.size):
                if self.winner_row:
                    if self.app.board[r,c] == 1:
                        self.draw_black(painter, (r, c), 1 if (r,c) in self.winner_row else 0.5)
                    elif self.app.board[r,c] == -1:
                        self.draw_white(painter, (r, c), 1 if (r,c) in self.winner_row else 0.5)
                else:
                    if self.app.board[r,c] == 1:
                        self.draw_black(painter, (r, c), 1)
                    elif self.app.board[r,c] == -1:
                        self.draw_white(painter, (r, c), 1)

    def draw_black(self, painter, intersection, opacity):
        painter.setBrush(QtGui.QBrush(QtGui.QColor(40, 40, 40)))
        painter.setOpacity(opacity)
        gap = self.frameGeometry().width() / self.app.size
        x = 5 + intersection[1] * gap
        y = 5 + intersection[0] * gap
        painter.drawEllipse(x, y, gap*0.75, gap*0.75)
    
    def draw_white(self, painter, intersection, opacity):
        painter.setBrush(QtGui.QBrush(QtGui.QColor(215, 215, 215))) 
        painter.setOpacity(opacity)
        gap = self.frameGeometry().width() / self.app.size
        x = 5 + intersection[1] * gap
        y = 5 + intersection[0] * gap
        painter.drawEllipse(x, y, gap*0.75, gap*0.75)

    def mousePressEvent(self, event):
        x = int(event.x() / 40)
        y = int(event.y() / 40)
        action = y*self.app.size + x
        self.app.player_step(action)

    def get_winner_row(self):
        for c in range(self.app.size-4):
            for r in range(self.app.size):
                if self.app.board[r,c] == self.app.board[r,c+1] == self.app.board[r,c+2] == self.app.board[r,c+3] == self.app.board[r,c+4] != 0:
                    return [(r, c), (r,c+1), (r,c+2), (r,c+3), (r,c+4)]

        for c in range(self.app.size):
            for r in range(self.app.size-4):
                if self.app.board[r,c] == self.app.board[r+1,c] == self.app.board[r+2,c] == self.app.board[r+3,c] == self.app.board[r+4,c] != 0:
                    return [(r, c), (r+1,c), (r+2,c), (r+3,c), (r+4,c)]

        for c in range(self.app.size-4):
            for r in range(self.app.size-4):
                if self.app.board[r,c] == self.app.board[r+1,c+1] == self.app.board[r+2,c+2] == self.app.board[r+3,c+3] == self.app.board[r+4,c+4] != 0:
                    return [(r, c), (r+1,c+1), (r+2,c+2), (r+3,c+3), (r+4,c+4)]

        for c in range(self.app.size-4):
            for r in range(4, self.app.size):
                if self.app.board[r,c] == self.app.board[r-1,c+1] == self.app.board[r-2,c+2] == self.app.board[r-3,c+3] == self.app.board[r-4,c+4] != 0:
                    return [(r, c), (r-1,c+1), (r-2,c+2), (r-3,c+3), (r-4,c+4)]
                
        return []
        