import numpy as np
from PyQt5 import QtGui, QtCore, QtWidgets

from .rules import Connect4Rules
from .network import Connect4Network
from ...mcts import MCTS
from ...misc import PrintColors as PC
from ...config import Config

class Connect4GUI(QtWidgets.QMainWindow):
    def __init__(self, rules: Connect4Rules, network: Connect4Network, config: Config):
        super().__init__()
        self.rules = rules
        self.network = network
        self.config = config
        self.mcts = MCTS(self.rules, self.network, self.config)

        self.network_turn = -1
        self.move = 1
        self.cur_player = 1
        self.winner = None
        self.state = self.rules.get_start_state()

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
        if self.cur_player == self.network_turn and self.winner is None:
            observation = self.state if self.cur_player == 1 else self.rules.flip_view(self.state)
            pi = self.mcts.get_policy(observation)
            action = np.argmax(pi)

            perceived_value = self.mcts.Q[(self.rules.hash(observation), action)]
            if not isinstance(perceived_value, float):
                perceived_value = perceived_value[0]
            perceived_value = perceived_value if self.network_turn == 1 else -perceived_value
            self.print_perceived_value(perceived_value)

            self.state = self.rules.step(self.state, action, self.cur_player)
            self.winner = self.rules.get_winner(self.state)
            self.cur_player *= -1
            self.connect4_widget.draw()

    def player_step(self, action: int) -> None:
        if self.winner is not None:
            self.state = self.rules.get_start_state()
            self.winner = None
            self.cur_player = 1
            self.network_turn *= -1
            self.mcts = MCTS(self.rules, self.network, self.config)
            self.move = 1
            self.connect4_widget.draw()
        else:
            if self.rules.get_valid_actions(self.state, self.cur_player)[action] and self.winner is None:
                self.state = self.rules.step(self.state, action, self.cur_player)
                self.winner = self.rules.get_winner(self.state)
                self.cur_player *= -1
                self.connect4_widget.draw()

    def keyPressEvent(self, event):
        key_press = event.key()

        if key_press == 16777216:
            self.timer.stop()
            exit()

    def print_perceived_value(self, perceived_value: float) -> None:
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
            v_color = PC.red
        else:
            v_color = PC.green

        p1_color = PC.green if self.network_turn == -1 else PC.red
        p2_color = PC.green if self.network_turn == 1 else PC.red

        evaluation_message = "AlphaZero is "
        if v_color == PC.green:
            evaluation_message += "worse "
        else:
            evaluation_message = evaluation_message + "better"

        move_len = len(str(self.move))
        move_suffix_space = "                           "
        move_suffix_space += " " * (4 - move_len)

        print(f"{PC.transparent}| ----------------------------------- |{PC.endc}")
        print(f"{PC.transparent}|{PC.endc} Move {self.move}{move_suffix_space}{PC.transparent}|{PC.endc}")
        print(f"{PC.transparent}|{PC.endc} AlphaZero perceived value : {v_color}{PC.bold}{perceived_str}{PC.endc} {PC.transparent}|{PC.endc}")
        print(f"{PC.transparent}|{PC.endc}                 {PC.transparent}{evaluation_message} |{PC.endc}")
        print(f"{PC.transparent}|{PC.endc}                                     {PC.transparent}|{PC.endc}")
        print(f"{PC.transparent}|{PC.endc}   AlphaZero perceived likelihoods   {PC.transparent}|{PC.endc}")
        print(f"{PC.transparent}|{PC.endc} {p1} ({PC.transparent}black{PC.endc})     win : {p1_color}{PC.bold}{lp1w}{PC.endc}% {PC.transparent}|{PC.endc}")
        print(f"{PC.transparent}|{PC.endc} {p2} ({PC.red}red{PC.endc})       win : {p2_color}{PC.bold}{lp2w}{PC.endc}% {PC.transparent}|{PC.endc}")
        print(f"{PC.transparent}|{PC.endc}                      Draw : {PC.bold}{ld}{PC.endc}% {PC.transparent}|{PC.endc}")
        print(f"{PC.transparent}| ----------------------------------- |{PC.endc}\n")

        self.move += 1
    
class Connect4Widget(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget, app: Connect4GUI):
        super().__init__(parent)
        self.app = app

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
                    if (y, x) in self.winner_row:# or self.app.state[y,x] == 0:
                        painter.setOpacity(1)
                    else:
                        painter.setOpacity(0.55)
                else:
                    painter.setOpacity(1)
                    
                if self.app.state[0, y, x] == 1:
                    painter.setBrush(QtGui.QBrush(QtGui.QColor(30, 30, 30)))
                elif self.app.state[1, y, x] == 1:
                    painter.setBrush(QtGui.QBrush(QtGui.QColor(225, 0, 0)))
                else:
                    painter.setBrush(QtGui.QBrush(QtCore.Qt.white))

                painter.drawEllipse(x_, y_, circle_size, circle_size)

            x_ = start_x + x*gap + 19

            painter.drawText(x_, 175 + 120 + 60, str(x+1))

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        x = min(int((event.x() + 5) / 55), 6)
        self.app.player_step(x)

    def get_winner_row(self) -> list[tuple[int, int]]:
        for i in range(2):
            for c in range(7):
                for r in range(6):
                    if c < 4:
                        if self.app.state[i,r,c] == self.app.state[i,r,c+1] == self.app.state[i,r,c+2] == self.app.state[i,r,c+3] == 1:
                            return [(r,c), (r,c+1), (r,c+2), (r,c+3)]
                    if r < 3:
                        if self.app.state[i,r,c] == self.app.state[i,r+1,c] == self.app.state[i,r+2,c] == self.app.state[i,r+3,c] == 1:
                            return [(r,c), (r+1,c), (r+2,c), (r+3,c)]
                    if c < 4 and r < 3:
                        if self.app.state[i,r,c] == self.app.state[i,r+1,c+1] == self.app.state[i,r+2,c+2] == self.app.state[i,r+3,c+3] == 1:
                            return [(r,c), (r+1,c+1), (r+2,c+2), (r+3,c+3)]
                    if c < 4 and r >= 3:
                        if self.app.state[i,r,c] == self.app.state[i,r-1,c+1] == self.app.state[i,r-2,c+2] == self.app.state[i,r-3,c+3] == 1:
                            return [(r,c), (r-1,c+1), (r-2,c+2), (r-3,c+3)]
        return []
