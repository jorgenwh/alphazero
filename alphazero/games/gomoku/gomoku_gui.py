import numpy as np
from PyQt5 import QtGui, QtCore, QtWidgets
from typing import List, Tuple

from .gomoku_rules import GomokuRules
from .gomoku_network import GomokuNetwork
from alphazero.mcts import MCTS
from alphazero.misc import Arguments, PrintColors

class GomokuGui(QtWidgets.QMainWindow):
    def __init__(self, rules: GomokuRules, network: GomokuNetwork, args: Arguments):
        super().__init__()
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), QtGui.QColor(255, 212, 129))
        self.setPalette(p)

        self.rules = rules
        self.network = network
        self.args = args
        self.size = self.args.gomoku_size

        # Build the MCTS that will provide the policy for AZ
        self.mcts = MCTS(self.rules, self.network, self.args)

        # Which player is the network currently playing as {1, -1}
        self.network_turn = -1

        self.move = 1
        self.cur_player = 1
        self.board = self.rules.get_start_board()

        self.init_window()
        self.fps = 60 
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.step)
        self.timer.start(1000 / self.fps)
        self.show()

    def init_window(self) -> None:
        self.setFixedSize(self.size*40 + 75, self.size*40 + 75)
        self.centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.centralWidget)
        self.setGeometry(1500, 330, 350, 350)
        self.gomoku_widget = GomokuWidget(self.centralWidget, self)
        self.gomoku_widget.setGeometry(40, 40, 40*self.size, 40*self.size)

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
            self.gomoku_widget.draw()

    def player_step(self, action: int) -> None:
        if self.rules.is_concluded(self.board):
            self.board = self.rules.get_start_board()
            self.cur_player = 1
            self.network_turn *= -1
            self.mcts = MCTS(self.rules, self.network, self.args)
            self.gomoku_widget.draw()
            self.move = 1
        else:
            if self.rules.get_valid_actions(self.board, self.cur_player)[action] and not self.rules.is_concluded(self.board):
                self.board = self.rules.step(self.board, action, self.cur_player)
                self.cur_player *= -1
                self.gomoku_widget.draw()

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

        if (self.network_turn == 1 and perceived_value > 0) or (self.network_turn == -1 and perceived_value < 0):
            v_color = PrintColors.red
        else:
            v_color = PrintColors.green

        p1_color = PrintColors.green if self.network_turn == -1 else PrintColors.red
        p2_color = PrintColors.green if self.network_turn == 1 else PrintColors.red

        evaluation_message = "AlphaZero is "
        if v_color == PrintColors.green:
            evaluation_message += "behind"
        else:
            evaluation_message = " " + evaluation_message + "ahead"

        print(f"{PrintColors.transparent}| ----------------------------------- |{PrintColors.endc}")
        print(f"{PrintColors.transparent}|{PrintColors.endc} Move {self.move}                              {PrintColors.transparent}|{PrintColors.endc}")
        print(f"{PrintColors.transparent}|{PrintColors.endc} AlphaZero perceived value : {v_color}{PrintColors.bold}{perceived_str}{PrintColors.endc} {PrintColors.transparent}|{PrintColors.endc}")
        print(f"{PrintColors.transparent}|{PrintColors.endc}                 {PrintColors.transparent}{evaluation_message} |{PrintColors.endc}")
        print(f"{PrintColors.transparent}|{PrintColors.endc}                                     {PrintColors.transparent}|{PrintColors.endc}")
        print(f"{PrintColors.transparent}|{PrintColors.endc}   AlphaZero perceived likelihoods   {PrintColors.transparent}|{PrintColors.endc}")
        print(f"{PrintColors.transparent}|{PrintColors.endc} {p1} ({PrintColors.transparent}black{PrintColors.endc})     win : {p1_color}{PrintColors.bold}{lp1w}{PrintColors.endc}% {PrintColors.transparent}|{PrintColors.endc}")
        print(f"{PrintColors.transparent}|{PrintColors.endc} {p2} (white)     win : {p2_color}{PrintColors.bold}{lp2w}{PrintColors.endc}% {PrintColors.transparent}|{PrintColors.endc}")
        print(f"{PrintColors.transparent}|{PrintColors.endc}                      Draw : {PrintColors.bold}{ld}{PrintColors.endc}% {PrintColors.transparent}|{PrintColors.endc}")
        print(f"{PrintColors.transparent}| ----------------------------------- |{PrintColors.endc}\n")

        self.move += 1

class GomokuWidget(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget, app: GomokuGui):
        super().__init__(parent)
        self.app = app

        # List to contain the 5 pieces making up the winning sequence.
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
        self.draw_stones(painter)
        painter.end()

    def draw_board(self, painter: QtGui.QPainter) -> None:
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

    def draw_stones(self, painter: QtGui.QPainter) -> None:
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

    def draw_black(self, painter: QtGui.QPainter, intersection: Tuple[int, int], opacity: float) -> None:
        painter.setBrush(QtGui.QBrush(QtGui.QColor(40, 40, 40)))
        painter.setOpacity(opacity)
        gap = self.frameGeometry().width() / self.app.size
        x = 5 + intersection[1] * gap
        y = 5 + intersection[0] * gap
        painter.drawEllipse(x, y, gap*0.75, gap*0.75)
    
    def draw_white(self, painter: QtGui.QPainter, intersection: Tuple[int, int], opacity: float) -> None:
        painter.setBrush(QtGui.QBrush(QtGui.QColor(215, 215, 215))) 
        painter.setOpacity(opacity)
        gap = self.frameGeometry().width() / self.app.size
        x = 5 + intersection[1] * gap
        y = 5 + intersection[0] * gap
        painter.drawEllipse(x, y, gap*0.75, gap*0.75)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        x = int(event.x() / 40)
        y = int(event.y() / 40)
        action = y*self.app.size + x
        self.app.player_step(action)

    def get_winner_row(self) -> List[Tuple[int, int]]:
        for c in range(self.app.size):
            for r in range(self.app.size):
                if c < self.app.size - 4:
                    if self.app.board[r,c] == self.app.board[r,c+1] == self.app.board[r,c+2] == self.app.board[r,c+3] == self.app.board[r,c+4] != 0:
                        return [(r, c), (r,c+1), (r,c+2), (r,c+3), (r,c+4)]

                if r < self.app.size - 4:
                    if self.app.board[r,c] == self.app.board[r+1,c] == self.app.board[r+2,c] == self.app.board[r+3,c] == self.app.board[r+4,c] != 0:
                        return [(r, c), (r+1,c), (r+2,c), (r+3,c), (r+4,c)]

                if c < self.app.size - 4 and r < self.app.size - 4:
                    if self.app.board[r,c] == self.app.board[r+1,c+1] == self.app.board[r+2,c+2] == self.app.board[r+3,c+3] == self.app.board[r+4,c+4] != 0:
                        return [(r, c), (r+1,c+1), (r+2,c+2), (r+3,c+3), (r+4,c+4)]
                        
                if c < self.app.size - 4 and r >= 4:
                    if self.app.board[r,c] == self.app.board[r-1,c+1] == self.app.board[r-2,c+2] == self.app.board[r-3,c+3] == self.app.board[r-4,c+4] != 0:
                        return [(r, c), (r-1,c+1), (r-2,c+2), (r-3,c+3), (r-4,c+4)]
        return []
