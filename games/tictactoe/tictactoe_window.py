from PyQt5 import QtGui, QtCore, QtWidgets
import numpy as np

class TicTacToeWindow(QtWidgets.QMainWindow):
    def __init__(self, game_rules, policy, args):
        super().__init__()
        self.game_rules = game_rules
        self.policy = policy
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
        self.setFixedSize(350, 350)
        self.centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.centralWidget)
        self.setGeometry(750, 280, 350, 350)

        self.tictactoe_widget = TicTacToeWidget(self.centralWidget, self)
        self.tictactoe_widget.setGeometry(40, 40, 270, 270)

    def step(self):
        if self.cur_player == self.nnet_turn and not self.game_rules.terminal(self.board):
            board_perspective = self.game_rules.perspective(self.board, self.cur_player)
            pi = self.policy.get_policy(board_perspective, t=0)
            action = np.argmax(pi)
            self.board, self.cur_player = self.game_rules.step(self.board, action, self.cur_player)
            self.tictactoe_widget.draw()

    def player_step(self, action):
        if self.game_rules.terminal(self.board):
            self.board = self.game_rules.start_board()
            self.cur_player = 1
            self.nnet_turn *= -1
            self.tictactoe_widget.draw()
        else:
            if self.game_rules.get_valid_actions(self.board, self.cur_player)[action] and not self.game_rules.terminal(self.board):
                self.board, self.cur_player = self.game_rules.step(self.board, action, self.cur_player)
                self.tictactoe_widget.draw()


class TicTacToeWidget(QtWidgets.QWidget):
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
        self.draw_signs(painter)
        painter.end()

    def draw_board(self, painter):
        painter.setRenderHints(QtGui.QPainter.Antialiasing)
        painter.setRenderHints(QtGui.QPainter.HighQualityAntialiasing)
        painter.setRenderHint(QtGui.QPainter.TextAntialiasing)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        painter.setPen(QtGui.QPen(QtCore.Qt.black))

        right = self.frameGeometry().width()
        bottom = self.frameGeometry().height()
        gap = right/3
        
        for i in range(2):
            painter.drawLine(0, 90 + i*gap, right, 90 + i*gap)
            painter.drawLine(90 + i*gap, 0, 90 + i*gap, bottom)

    def draw_signs(self, painter):
        for r in range(self.app.board.shape[0]):
            for c in range(self.app.board.shape[1]):

                if self.winner_row:
                    if (r, c) in self.winner_row or self.app.board[r,c] == 0:
                        painter.setOpacity(1)
                    else:
                        painter.setOpacity(0.5)
                else:
                    painter.setOpacity(1)

                if self.app.board[r, c] == 1:
                    self.draw_cross(painter, (r, c))
                elif self.app.board[r, c] == -1:
                    self.draw_circle(painter, (r, c))

    def draw_cross(self, painter, cell):
        gap = self.frameGeometry().width()/3
        painter.drawLine(cell[1]*gap + gap/2 - 25, cell[0]*gap + gap/2 - 25, cell[1]*gap + gap/2 + 25, cell[0]*gap + gap/2 + 25)
        painter.drawLine(cell[1]*gap + gap/2 - 25, cell[0]*gap + gap/2 + 25, cell[1]*gap + gap/2 + 25, cell[0]*gap + gap/2 - 25)

    def draw_circle(self, painter, cell):
        gap = self.frameGeometry().width()/3
        painter.drawEllipse(cell[1]*gap + 15, cell[0]*gap + 15, 60, 60)

    def mousePressEvent(self, event):
        r = int(event.x() / 90)
        c = int(event.y() / 90)
        action = c*3 + r
        self.app.player_step(action)

    def get_winner_row(self):
        for r in range(3):
            if self.app.board[r,0] == self.app.board[r,1] == self.app.board[r,2] != 0:
                return [(r,0), (r,1), (r,2)]
        
        for c in range(3):
            if self.app.board[0,c] == self.app.board[1,c] == self.app.board[2,c] != 0:
                return [(0,c), (1,c), (2,c)]

        if self.app.board[0,0] == self.app.board[1,1] == self.app.board[2,2] != 0:
            return [(0,0), (1,1), (2,2)]
        
        if self.app.board[2,0] == self.app.board[1,1] == self.app.board[0,2] != 0:
            return [(2,0), (1,1), (0,2)]
        
        return []
        