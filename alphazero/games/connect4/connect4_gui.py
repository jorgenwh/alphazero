import numpy as np
from PyQt5 import QtGui, QtCore, QtWidgets
from typing import List, Tuple

from .connect4_rules import Connect4Rules
from .connect4_network import Connect4Network
from alphazero.mcts import MCTS
from alphazero.misc import Arguments

class Connect4Gui(QtWidgets.QMainWindow):
    def __init__(self, rules: Connect4Rules, network: Connect4Network, args: Arguments):
        super().__init__()
        self.rules = rules
        self.network = network
        self.args = args

        # Build the MCTS that will provide the policy from AZ
        self.mcts = MCTS(self.rules, self.network, self.args)

        # Which player is the network currently playing as {1, -1}
        self.network_turn = -1

        self.cur_player = 1
        self.board = self.rules.get_start_board()

        self.init_window()
        self.fps = 60 
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.step)
        self.timer.start(1000 / self.fps)
        self.show()

    def init_window(self) -> None:
        self.setFixedSize((60*7) - 5, (60*6) + 50)
        self.centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.centralWidget)
        self.setGeometry(750, 280, (50*7) - 5, (50*6) + 50)

        self.connect4_widget = Connect4Widget(self.centralWidget, self)
        self.connect4_widget.setGeometry(20, 20, 45*7 + 10*6, 60*6)

    def step(self) -> None:
        if self.cur_player == self.network_turn and not self.rules.is_concluded(self.board):
            perspective = self.board if self.cur_player == 1 else self.rules.flip(self.board)
            pi = self.mcts.get_policy(perspective, temperature=0)
            action = np.argmax(pi)

            perceived_value = self.mcts.Q[(self.rules.to_string(perspective), action)]
            if not isinstance(perceived_value, float):
                perceived_value = perceived_value[0]
            perceived_value = perceived_value if self.network_turn == 1 else -perceived_value
            self.print_perception(perceived_value)

            self.board = self.rules.step(self.board, action, self.cur_player)
            self.cur_player *= -1
            self.connect4_widget.draw()

    def player_step(self, action: int) -> None:
        if self.rules.is_concluded(self.board):
            self.board = self.rules.get_start_board()
            self.cur_player = 1
            self.network_turn *= -1
            self.mcts = MCTS(self.rules, self.network, self.args)
            self.connect4_widget.draw()
        else:
            if self.rules.get_valid_actions(self.board, self.cur_player)[action] and not self.rules.is_concluded(self.board):
                self.board = self.rules.step(self.board, action, self.cur_player)
                self.cur_player *= -1
                self.connect4_widget.draw()

    def print_perception(self, perceived_value: float) -> None:
        perceived_str = str(round(perceived_value, 4))
        lp1w = str(round(max(perceived_value, 0) * 100, 1))
        lp2w = str(round(max(-perceived_value, 0) * 100, 1))
        ld = str(round((1.0 - abs(perceived_value)) * 100, 1))

        perceived_str = " " + perceived_str if len(perceived_str) == 6 else perceived_str
        perceived_str += " " * (7 - len(perceived_str))
        lp1w += " " * (6 - len(lp1w))
        lp2w += " " * (6 - len(lp2w))
        ld += " " * (6 - len(ld))

        p1 = "AlphaZero" if self.network_turn == 1 else "Human    "
        p2 = "AlphaZero" if self.network_turn == -1 else "Human    "

        print(f"| ----------------------------------- |")
        print(f"| AlphaZero perceived value : {perceived_str} |")
        print(f"|                                     |")
        print(f"|   AlphaZero perceived likelihoods   |")
        print(f"| {p1}  (black) win    : {lp1w}% |")
        print(f"| {p2}  (white) win    : {lp2w}% |")
        print(f"| Draw                      : {ld}% |")
        print(f"| ----------------------------------- |\n")

class Connect4Widget(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget, app: Connect4Gui):
        super().__init__(parent)
        self.app = app

        # List to contain the 4 pieces making up the winning sequence.
        # This is used to color the 4 pieces in the winning sequence differently when the game is concluded
        self.winner_row = []
        self.show()

    def draw(self) -> None:
        self.repaint()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter()
        self.winner_row = self.get_winner_row()
        painter.begin(self)
        self.draw_board(painter)
        painter.end()

    def draw_board(self, painter: QtGui.QPainter) -> None:
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

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        x = min(int((event.x() + 5) / 55), 6)
        self.app.player_step(x)

    def get_winner_row(self) -> List[Tuple[int, int]]:
        for c in range(7):
            for r in range(6):
                if c < 4:
                    if self.app.board[r,c] == self.app.board[r,c+1] == self.app.board[r,c+2] == self.app.board[r,c+3] != 0:
                        return [(r,c), (r,c+1), (r,c+2), (r,c+3)]
                if r < 3:
                    if self.app.board[r,c] == self.app.board[r+1,c] == self.app.board[r+2,c] == self.app.board[r+3,c] != 0:
                        return [(r,c), (r+1,c), (r+2,c), (r+3,c)]
                if c < 4 and r < 3:
                    if self.app.board[r,c] == self.app.board[r+1,c+1] == self.app.board[r+2,c+2] == self.app.board[r+3,c+3] != 0:
                        return [(r,c), (r+1,c+1), (r+2,c+2), (r+3,c+3)]
                if c < 4 and r >= 3:
                    if self.app.board[r,c] == self.app.board[r-1,c+1] == self.app.board[r-2,c+2] == self.app.board[r-3,c+3] != 0:
                        return [(r,c), (r-1,c+1), (r-2,c+2), (r-3,c+3)]
        return []
