from PyQt5 import QtGui, QtCore, QtWidgets
import numpy as np

class Othello_Window(QtWidgets.QMainWindow):
    def __init__(self, game_rules, mcts, args):
        super().__init__()
        self.game_rules = game_rules
        self.mcts = mcts
        self.args = args
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
        self.setFixedSize((60*7) - 5, (60*6) + 50)
        self.centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.centralWidget)
        self.setGeometry(750, 280, (50*7) - 5, (50*6) + 50)

        self.othello_widget = Othello_Widget(self.centralWidget, self)
        self.othello_widget.setGeometry(20, 20, 45*7 + 10*6, 60*6)

    def step(self):
        if self.cur_player == self.nnet_turn and not self.game_rules.terminal(self.board):
            board_perspective = self.game_rules.perspective(self.board, self.cur_player)
            self.mcts.tree_search(board_perspective)
            pi = self.mcts.get_policy(board_perspective, t=0)
            action = np.argmax(pi)
            self.board, self.cur_player = self.game_rules.step(self.board, action, self.cur_player)
            self.othello_widget.draw()

    def player_step(self, action):
        if self.game_rules.terminal(self.board):
            self.board = self.game_rules.start_board()
            self.cur_player = 1
            self.nnet_turn *= -1
            self.othello_widget.draw()
        else:
            if self.game_rules.get_valid_actions(self.board)[action] and not self.game_rules.terminal(self.board):
                self.board, self.cur_player = self.game_rules.step(self.board, action, self.cur_player)
                self.othello_widget.draw()


class Othello_Widget(QtWidgets.QWidget):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.show()

    def draw(self):
        self.repaint()

    def paintEvent(self, event):
        painter = QtGui.QPainter()
        painter.begin(self)
        self.draw_board(painter)
        painter.end()

    def draw_board(self, painter):
        painter.setRenderHints(QtGui.QPainter.Antialiasing)
        painter.setRenderHints(QtGui.QPainter.HighQualityAntialiasing)
        painter.setRenderHint(QtGui.QPainter.TextAntialiasing)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        painter.setPen(QtGui.QPen(QtCore.Qt.black))
        
        font = QtGui.QFont("Helvetica", 12)
        painter.setFont(font)
        painter.setPen(QtCore.Qt.black)

        circle_size = 45
        start_x = 0
        start_y = 15
        gap = 55

        for x in range(7):
            for y in range(6):
                x_ = start_x + x*gap
                y_ = start_y + y*gap
                    
                if self.app.board[y, x] == 1:
                    painter.setBrush(QtGui.QBrush(QtGui.QColor(30, 30, 30)))
                elif self.app.board[y, x] == -1:
                    painter.setBrush(QtGui.QBrush(QtGui.QColor(225, 0, 0)))
                else:
                    painter.setBrush(QtGui.QBrush(QtCore.Qt.white))

                painter.drawEllipse(x_, y_, circle_size, circle_size)


    def mousePressEvent(self, event):
        pass
        self.app.player_step(x)