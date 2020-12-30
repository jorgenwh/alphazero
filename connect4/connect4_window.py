from PyQt5 import QtGui, QtCore, QtWidgets
import numpy as np

class Connect4_Window(QtWidgets.QMainWindow):
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

        self.connect4_widget = Connect4_Widget(self.centralWidget, self)
        self.connect4_widget.setGeometry(20, 20, 45*7 + 10*6, 60*6)

    def step(self):
        if self.cur_player == self.nnet_turn and not self.game_rules.terminal(self.board):
            board_perspective = self.game_rules.perspective(self.board, self.cur_player)
            pi = self.mcts.tree_search(board_perspective, temperature=0)
            action = np.argmax(pi)
            self.board, self.cur_player = self.game_rules.step(self.board, action, self.cur_player)
            self.connect4_widget.draw()

    def player_step(self, action):
        if self.game_rules.terminal(self.board):
            self.board = self.game_rules.start_board()
            self.cur_player = 1
            self.nnet_turn *= -1
            self.connect4_widget.draw()
        else:
            if self.game_rules.get_valid_actions(self.board)[action] and not self.game_rules.terminal(self.board):
                self.board, self.cur_player = self.game_rules.step(self.board, action, self.cur_player)
                self.connect4_widget.draw()


class Connect4_Widget(QtWidgets.QWidget):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.winner_row = []
        self.show()

    def draw(self):
        self.repaint()

    def paintEvent(self, event):
        painter = QtGui.QPainter()
        self.winner_row = self.get_winner_row()
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

                if self.winner_row:
                    if (y, x) in self.winner_row or self.app.board[y,x] == 0:
                        painter.setOpacity(1)
                    else:
                        painter.setOpacity(0.55)
                else:
                    painter.setOpacity(1)
                    
                if self.app.board[y, x] == 1:
                    painter.setBrush(QtGui.QBrush(QtGui.QColor(30, 30, 30)))
                elif self.app.board[y, x] == -1:
                    painter.setBrush(QtGui.QBrush(QtGui.QColor(225, 0, 0)))
                else:
                    painter.setBrush(QtGui.QBrush(QtCore.Qt.white))

                painter.drawEllipse(x_, y_, circle_size, circle_size)

            x_ = start_x + x*gap + 19

            painter.drawText(x_, 175 + 120 + 60, str(x+1))

    def mousePressEvent(self, event):
        x = min(int((event.x() + 5) / 55), 6)
        self.app.player_step(x)

    def get_winner_row(self):
        for c in range(7-3):
            for r in range(6):
                if self.app.board[r,c] == self.app.board[r,c+1] == self.app.board[r,c+2] == self.app.board[r,c+3] != 0:
                    return [(r,c), (r,c+1), (r,c+2), (r,c+3)]

        for c in range(7):
            for r in range(6-3):
                if self.app.board[r,c] == self.app.board[r+1,c] == self.app.board[r+2,c] == self.app.board[r+3,c] != 0:
                    return [(r,c), (r+1,c), (r+2,c), (r+3,c)]
     
        for c in range(7-3):
            for r in range(6-3):
                if self.app.board[r,c] == self.app.board[r+1,c+1] == self.app.board[r+2,c+2] == self.app.board[r+3,c+3] != 0:
                    return [(r,c), (r+1,c+1), (r+2,c+2), (r+3,c+3)]

        for c in range(7-3):
            for r in range(3, 6):
                if self.app.board[r,c] == self.app.board[r-1,c+1] == self.app.board[r-2,c+2] == self.app.board[r-3,c+3] != 0:
                    return [(r,c), (r-1,c+1), (r-2,c+2), (r-3,c+3)]

        return []