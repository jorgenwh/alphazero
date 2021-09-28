import numpy as np
from PyQt5 import QtGui, QtCore, QtWidgets
from typing import List, Tuple

from .tictactoe_rules import TicTacToeRules
from .tictactoe_network import TicTacToeNetwork
from alphazero.mcts import MCTS
from alphazero.misc import Arguments

class TicTacToeGui(QtWidgets.QMainWindow):
    def __init__(self, rules: TicTacToeRules, network: TicTacToeNetwork, args: Arguments):
        super().__init__()
        self.rules = rules
        self.network = network
        self.args = args

        # Build the MCTS that will provide the policy for AZ
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
        self.setFixedSize(350, 350)
        self.centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.centralWidget)
        self.setGeometry(750, 280, 350, 350)

        self.tictactoe_widget = TicTacToeWidget(self.centralWidget, self)
        self.tictactoe_widget.setGeometry(40, 40, 270, 270)

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
            self.tictactoe_widget.draw()

    def player_step(self, action: int) -> None:
        if self.rules.is_concluded(self.board):
            self.board = self.rules.get_start_board()
            self.cur_player = 1
            self.network_turn *= -1
            self.mcts = MCTS(self.rules, self.network, self.args)
            self.tictactoe_widget.draw()
        else:
            if self.rules.get_valid_actions(self.board, self.cur_player)[action] and not self.rules.is_concluded(self.board):
                self.board = self.rules.step(self.board, action, self.cur_player)
                self.cur_player *= -1
                self.tictactoe_widget.draw()

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


class TicTacToeWidget(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget, app: TicTacToeGui):
        super().__init__(parent)
        self.app = app

        # List to contain the 3 slots making up the winning sequence.
        # This is used to color the winning sequence differently when the game is concluded
        self.winner_row = []
        self.show()

    def draw(self) -> None:
        self.repaint()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter()
        painter.begin(self)
        self.winner_row = self.get_winner_row()
        self.draw_board(painter)
        self.draw_signs(painter)
        painter.end()

    def draw_board(self, painter: QtGui.QPainter) -> None:
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

    def draw_signs(self, painter: QtGui.QPainter) -> None:
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

    def draw_cross(self, painter: QtGui.QPainter, cell: Tuple[int, int]) -> None:
        gap = self.frameGeometry().width()/3
        painter.drawLine(cell[1]*gap + gap/2 - 25, cell[0]*gap + gap/2 - 25, cell[1]*gap + gap/2 + 25, cell[0]*gap + gap/2 + 25)
        painter.drawLine(cell[1]*gap + gap/2 - 25, cell[0]*gap + gap/2 + 25, cell[1]*gap + gap/2 + 25, cell[0]*gap + gap/2 - 25)

    def draw_circle(self, painter: QtGui.QPainter, cell: Tuple[int, int]) -> None:
        gap = self.frameGeometry().width()/3
        painter.drawEllipse(cell[1]*gap + 15, cell[0]*gap + 15, 60, 60)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        r = int(event.x() / 90)
        c = int(event.y() / 90)
        action = c*3 + r
        self.app.player_step(action)

    def get_winner_row(self) -> List[Tuple[int, int]]:
        for i in range(3):
            if self.app.board[i,0] == self.app.board[i,1] == self.app.board[i,2] != 0:
                return [(i,0), (i,1), (i,2)]
            if self.app.board[0,i] == self.app.board[1,i] == self.app.board[2,i] != 0:
                return [(0,i), (1,i), (2,i)]

        if self.app.board[0,0] == self.app.board[1,1] == self.app.board[2,2] != 0:
            return [(0,0), (1,1), (2,2)]
        if self.app.board[2,0] == self.app.board[1,1] == self.app.board[0,2] != 0:
            return [(2,0), (1,1), (0,2)]
        
        return []
