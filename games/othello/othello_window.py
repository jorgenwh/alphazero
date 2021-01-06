from PyQt5 import QtGui, QtCore, QtWidgets
import numpy as np

class Othello_Window(QtWidgets.QMainWindow):
    def __init__(self, game_rules, mcts, args):
        super().__init__()
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), QtGui.QColor(80, 40, 40))
        self.setPalette(p)

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
        self.setFixedSize(65*8 + 50, 65*8 + 50)
        self.centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.centralWidget)
        self.setGeometry(750, 280, 65*8 + 50, 65*8 + 50)

        self.othello_widget = Othello_Widget(self.centralWidget, self)
        self.othello_widget.setGeometry(25, 25, 65*8, 65*8)

    def step(self):
        if self.cur_player == self.nnet_turn and not self.game_rules.terminal(self.board):
            board_perspective = self.game_rules.perspective(self.board, self.cur_player)
            self.mcts.tree_search(board_perspective)
            pi = self.mcts.get_policy(board_perspective, t=0)
            action = np.argmax(pi)
            self.board, self.cur_player = self.game_rules.step(self.board, action, self.cur_player)
            self.othello_widget.draw()
        elif not self.game_rules.terminal(self.board):
            if sum(self.game_rules.get_valid_actions(self.board, -self.nnet_turn)) == 0:
                self.board, self.cur_player = self.game_rules.step(self.board, 0, self.cur_player)
                self.othello_widget.draw()

    def player_step(self, action):
        if self.game_rules.terminal(self.board):
            self.board = self.game_rules.start_board()
            self.cur_player = 1
            self.nnet_turn *= -1
            self.othello_widget.draw()
        else:
            if self.game_rules.get_valid_actions(self.board, self.cur_player)[action] and not self.game_rules.terminal(self.board):
                self.board, self.cur_player = self.game_rules.step(self.board, action, self.cur_player)
                self.othello_widget.draw()


class Othello_Widget(QtWidgets.QWidget):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), QtGui.QColor(60, 130, 60))
        self.setPalette(p)
        self.winner = None

        self.app = app
        self.show()

    def draw(self):
        if self.app.game_rules.terminal(self.app.board):
            self.winner = self.app.game_rules.result(self.app.board, 1)
        else:
            self.winner = None

        self.repaint()

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
        gap = right / 8

        for i in range(9):
            painter.drawLine(i*gap, 0, i*gap, right)
            painter.drawLine(0, i*gap, bottom, i*gap)

    def draw_stones(self, painter):
        valid_actions = self.app.game_rules.get_valid_actions(self.app.board, self.app.cur_player)

        for r in range(8):
            for c in range(8):
                if self.app.board[r,c] == 1:
                    if self.winner == -1 or self.winner == 0:
                        self.draw_black(painter, (r, c), 0.5)
                    else:
                        self.draw_black(painter, (r, c), 1)
                elif self.app.board[r,c] == -1:
                    if self.winner == 1 or self.winner == 0:
                        self.draw_white(painter, (r, c), 0.5)
                    else:
                        self.draw_white(painter, (r, c), 1)

                if valid_actions[r * 8 + c]:
                    if self.app.cur_player == 1:
                        self.draw_black(painter, (r, c), 0.2)
                    elif self.app.cur_player == -1:
                        self.draw_white(painter, (r, c), 0.2)
                
    def draw_black(self, painter, intersection, opacity):
        painter.setBrush(QtGui.QBrush(QtGui.QColor(40, 40, 40)))
        painter.setOpacity(opacity)
        gap = self.frameGeometry().width() / 8
        x = 8.5 + intersection[1] * gap
        y = 8.5 + intersection[0] * gap
        painter.drawEllipse(x, y, gap*0.75, gap*0.75)
    
    def draw_white(self, painter, intersection, opacity):
        painter.setBrush(QtGui.QBrush(QtGui.QColor(215, 215, 215))) 
        painter.setOpacity(opacity)
        gap = self.frameGeometry().width() / 8
        x = 8.5 + intersection[1] * gap
        y = 8.5 + intersection[0] * gap
        painter.drawEllipse(x, y, gap*0.75, gap*0.75)

    def mousePressEvent(self, event):
        x = int(event.x() / 65)
        y = int(event.y() / 65)
        action = y*8 + x
        self.app.player_step(action)
