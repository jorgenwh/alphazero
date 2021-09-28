import numpy as np
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtGui import QFont
from typing import List, Tuple

from .othello_rules import OthelloRules
from .othello_network import OthelloNetwork
from alphazero.mcts import MCTS
from alphazero.misc import Arguments

class OthelloGui(QtWidgets.QMainWindow):
    def __init__(self, rules: OthelloRules, network: OthelloNetwork, args: Arguments):
        super().__init__()
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), QtGui.QColor(80, 40, 40))
        self.setPalette(p)

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
        self.setFixedSize(65*8 + 50, 65*8 + 50)
        self.centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.centralWidget)
        self.setGeometry(750, 280, 65*8 + 50, 65*8 + 50)

        self.othello_widget = OthelloWidget(self.centralWidget, self)
        self.othello_widget.setGeometry(25, 25, 65*8, 65*8)

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
            self.othello_widget.draw()

        elif not self.rules.is_concluded(self.board):
            if sum(self.rules.get_valid_actions(self.board, -self.network_turn)) == 0:
                self.board = self.rules.step(self.board, 0, self.cur_player)
                self.cur_player *= -1
                self.othello_widget.draw()

    def player_step(self, action: int) -> None:
        if self.rules.is_concluded(self.board):
            self.board = self.rules.get_start_board()
            self.cur_player = 1
            self.network_turn *= -1
            self.othello_widget.draw()
        else:
            if self.rules.get_valid_actions(self.board, self.cur_player)[action] and not self.rules.is_concluded(self.board):
                self.board = self.rules.step(self.board, action, self.cur_player)
                self.cur_player *= -1
                self.othello_widget.draw()

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

class OthelloWidget(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget, app: OthelloGui):
        super().__init__(parent)
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), QtGui.QColor(60, 130, 60))
        self.setPalette(p)
        self.app = app
        self.winner = None
        self.show()

    def draw(self):
        if self.app.rules.is_concluded(self.app.board):
            self.winner = self.app.rules.get_result(self.app.board)
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

    def get_scores(self, board: np.ndarray) -> Tuple[int, int]:
        black = white = 0
        for r in range(8):
            for c in range(8):
                if board[r,c] == 1:
                    black += 1
                elif board[r,c] == -1:
                    white += 1

        return black, white

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter()
        painter.begin(self)
        self.draw_board(painter)
        self.draw_stones(painter)
        painter.end()

    def draw_board(self, painter: QtGui.QPainter) -> None:
        painter.setRenderHints(QtGui.QPainter.Antialiasing)
        painter.setRenderHints(QtGui.QPainter.HighQualityAntialiasing)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        painter.setPen(QtGui.QPen(QtCore.Qt.black))

        right = self.frameGeometry().width()
        bottom = self.frameGeometry().height()
        gap = right / 8

        for i in range(8 + 1):
            painter.drawLine(i*gap, 0, i*gap, right)
            painter.drawLine(0, i*gap, bottom, i*gap)

    def draw_stones(self, painter: QtGui.QPainter) -> None:
        valid_actions = self.app.rules.get_valid_actions(self.app.board, self.app.cur_player)

        for r in range(8):
            for c in range(8):
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

                if valid_actions[r * 8 + c]:
                    if self.app.cur_player == 1:
                        self.draw_black(painter, (r, c), 0.2)
                    elif self.app.cur_player == -1:
                        self.draw_white(painter, (r, c), 0.2)
                
    def draw_black(self, painter: QtGui.QPainter, intersection: Tuple[int, int], opacity: float) -> None:
        painter.setBrush(QtGui.QBrush(QtGui.QColor(40, 40, 40)))
        painter.setOpacity(opacity)
        gap = self.frameGeometry().width() / 8
        x = 8.5 + intersection[1] * gap
        y = 8.5 + intersection[0] * gap
        painter.drawEllipse(x, y, gap*0.75, gap*0.75)
    
    def draw_white(self, painter: QtGui.QPainter, intersection: Tuple[int, int], opacity: float) -> None:
        painter.setBrush(QtGui.QBrush(QtGui.QColor(215, 215, 215))) 
        painter.setOpacity(opacity)
        gap = self.frameGeometry().width() / 8
        x = 8.5 + intersection[1] * gap
        y = 8.5 + intersection[0] * gap
        painter.drawEllipse(x, y, gap*0.75, gap*0.75)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        x = int(event.x() / 65)
        y = int(event.y() / 65)
        action = y*8 + x
        self.app.player_step(action)
