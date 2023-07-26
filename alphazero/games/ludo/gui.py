import numpy as np
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtGui import QPolygon, QPen, QColor
from PyQt5.QtCore import QPoint

from .rules import LudoRules
from .network import LudoNetwork
from ...mcts import MCTS
from ...misc import PrintColors as PC
from ...args import PLAY_TEMPERATURE

RED = QtGui.QColor(255, 75, 75)
RED_PIECE = QtGui.QColor(255, 0, 0)
GREEN = QtGui.QColor(75, 255, 75)
GREEN_PIECE = QtGui.QColor(0, 255, 0)
BLUE = QtGui.QColor(75, 75, 255)
BLUE_PIECE = QtGui.QColor(0, 0, 255)
YELLOW = QtGui.QColor(255, 255, 75)
YELLOW_PIECE = QtGui.QColor(255, 255, 0)

RED_INDEX_TO_POS = [
        None, 
        (1*60, 6*60), (2*60, 6*60), (3*60, 6*60), (4*60, 6*60), (5*60, 6*60), 
        (6*60, 5*60), (6*60, 4*60), (6*60, 3*60), (6*60, 2*60), (6*60, 1*60), (6*60, 0),
        (7*60, 0*60),
        (8*60, 0*60), (8*60, 1*60), (8*60, 2*60), (8*60, 3*60), (8*60, 4*60), (8*60, 5*60),
        (9*60, 6*60), (10*60, 6*60), (11*60, 6*60), (12*60, 6*60), (13*60, 6*60), (14*60, 6*60),
        (14*60, 7*60),
        (14*60, 8*60), (13*60, 8*60), (12*60, 8*60), (11*60, 8*60), (10*60, 8*60), (9*60, 8*60),
        (8*60, 9*60), (8*60, 10*60), (8*60, 11*60), (8*60, 12*60), (8*60, 13*60), (8*60, 14*60),
        (7*60, 14*60),
        (6*60, 14*60), (6*60, 13*60), (6*60, 12*60), (6*60, 11*60), (6*60, 10*60), (6*60, 9*60),
        (5*60, 8*60), (4*60, 8*60), (3*60, 8*60), (2*60, 8*60), (1*60, 8*60), (0*60, 8*60),
        (0*60, 7*60),
        (1*60, 7*60), (2*60, 7*60), (3*60, 7*60), (4*60, 7*60), (5*60, 7*60)
]

YELLOW_INDEX_TO_POS = [
        None, 
        (13*60, 8*60), (12*60, 8*60), (11*60, 8*60), (10*60, 8*60), (9*60, 8*60), 
        (8*60, 9*60), (8*60, 10*60), (8*60, 11*60), (8*60, 12*60), (8*60, 13*60), (8*60, 14*60),
        (7*60, 14*60),
        (6*60, 14*60), (6*60, 13*60), (6*60, 12*60), (6*60, 11*60), (6*60, 10*60), (6*60, 9*60),
        (5*60, 8*60), (4*60, 8*60), (3*60, 8*60), (2*60, 8*60), (1*60, 8*60), (0*60, 8*60),
        (0*60, 7*60),
        (0*60, 6*60), (1*60, 6*60), (2*60, 6*60), (3*60, 6*60), (4*60, 6*60), (5*60, 6*60),
        (6*60, 5*60), (6*60, 4*60), (6*60, 3*60), (6*60, 2*60), (6*60, 1*60), (6*60, 0*60),
        (7*60, 0*60),
        (8*60, 0*60), (8*60, 1*60), (8*60, 2*60), (8*60, 3*60), (8*60, 4*60), (8*60, 5*60),
        (9*60, 6*60), (10*60, 6*60), (11*60, 6*60), (12*60, 6*60), (13*60, 6*60), (14*60, 6*60),
        (14*60, 7*60),
        (13*60, 7*60), (12*60, 7*60), (11*60, 7*60), (10*60, 7*60), (9*60, 7*60)
]

class LudoGUI(QtWidgets.QMainWindow):
    def __init__(self, rules: LudoRules, network: LudoNetwork):
        super().__init__()
        self.rules = rules
        self.network = network
        self.mcts = MCTS(self.rules, self.network)

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
        self.setFixedSize(900, 900)
        self.centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.centralWidget)
        self.setGeometry(0, 0, 900, 900)
        self.ludo_widget = LudoWidget(self.centralWidget, self)
        self.ludo_widget.setGeometry(0, 0, 900, 900)

    def step(self) -> None:
        if self.cur_player == self.network_turn and self.winner is None:
            observation = self.state if self.cur_player == 1 else self.rules.flip_view(self.state)
            pi = self.mcts.get_policy(observation, temperature=PLAY_TEMPERATURE)
            action = np.argmax(pi)

            perceived_value = self.mcts.Q[(self.rules.hash(observation), action)]
            if not isinstance(perceived_value, float):
                perceived_value = perceived_value[0]
            perceived_value = perceived_value if self.network_turn == 1 else -perceived_value
            self.print_perceived_value(perceived_value)

            self.state = self.rules.step(self.state, action, self.cur_player)
            self.winner = self.rules.get_winner(self.state)
            self.cur_player *= -1
            self.ludo_widget.draw()

    def player_step(self, action: int) -> None:
        if self.winner is not None:
            self.state = self.rules.get_start_state()
            self.winner = None
            self.cur_player = 1
            self.network_turn *= -1
            self.mcts = MCTS(self.rules, self.network)
            self.move = 1
            self.ludo_widget.draw()
        else:
            if self.rules.get_valid_actions(self.state, self.cur_player)[action] and self.winner is None:
                self.state = self.rules.step(self.state, action, self.cur_player)
                self.winner = self.rules.get_winner(self.state)
                self.cur_player *= -1
                self.ludo_widget.draw()

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
    
class LudoWidget(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget, app: LudoGUI):
        super().__init__(parent)
        self.app = app
        self.show()

    def draw(self) -> None:
        self.repaint()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter()
        painter.begin(self)
        self.draw_board(painter)
        self.draw_pieces(painter)
        painter.end()

    def draw_board(self, painter: QtGui.QPainter) -> None:
        painter.setRenderHints(QtGui.QPainter.Antialiasing)
        painter.setRenderHints(QtGui.QPainter.HighQualityAntialiasing)
        painter.setRenderHint(QtGui.QPainter.TextAntialiasing)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        painter.setPen(QtGui.QPen(QtCore.Qt.black))
        
        font = QtGui.QFont("Helvetica", 12)
        painter.setFont(font)

        # color
        # red
        painter.setBrush(QtGui.QBrush(RED))
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawRect(60, 6*60, 60, 60)
        for i in range(1, 6):
            painter.drawRect(i*60, 7*60, 60, 60)
        for i in range(6):
            painter.drawRect(0, i*60, 60, 60)
            painter.drawRect(i*60, 0, 60, 60)
            painter.drawRect(5*60, i*60, 60, 60)
            painter.drawRect(i*60, 5*60, 60, 60)
        triangle = QPolygon([QPoint(6*60, 6*60), QPoint(450, 450), QPoint(6*60, 9*60)])
        painter.drawPolygon(triangle)

        # green
        painter.setBrush(QtGui.QBrush(GREEN))
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawRect(8*60, 1*60, 60, 60)
        for i in range(1, 6):
            painter.drawRect(7*60, i*60, 60, 60)
        for i in range(6):
            painter.drawRect(9*60, i*60, 60, 60)
            painter.drawRect(14*60, i*60, 60, 60)
            painter.drawRect(9*60 + i*60, 0, 60, 60)
            painter.drawRect(9*60 + i*60, 5*60, 60, 60)
        triangle = QPolygon([QPoint(6*60, 6*60), QPoint(450, 450), QPoint(9*60, 6*60)])
        painter.drawPolygon(triangle)

        # blue
        painter.setBrush(QtGui.QBrush(BLUE))
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawRect(6*60, 13*60, 60, 60)
        for i in range(1, 6):
            painter.drawRect(7*60, 8*60 + i*60, 60, 60)
        for i in range(6):
            painter.drawRect(0, 9*60 + i*60, 60, 60)
            painter.drawRect(5*60, 9*60 + i*60, 60, 60)
            painter.drawRect(i*60, 9*60, 60, 60)
            painter.drawRect(i*60, 14*60, 60, 60)
        triangle = QPolygon([QPoint(6*60, 9*60), QPoint(450, 450), QPoint(9*60, 9*60)])
        painter.drawPolygon(triangle)

        # yellow
        painter.setBrush(QtGui.QBrush(YELLOW))
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawRect(13*60, 8*60, 60, 60)
        for i in range(1, 6):
            painter.drawRect(8*60 + i*60, 7*60, 60, 60)
        for i in range(6):
            painter.drawRect(9*60 + i*60, 9*60, 60, 60)
            painter.drawRect(9*60 + i*60, 14*60, 60, 60)
            painter.drawRect(9*60, 9*60 + i*60, 60, 60)
            painter.drawRect(14*60, 9*60 + i*60, 60, 60)
        triangle = QPolygon([QPoint(9*60, 6*60), QPoint(450, 450), QPoint(9*60, 9*60)])
        painter.drawPolygon(triangle)

        painter.setPen(QtCore.Qt.black)
        painter.setBrush(QtCore.Qt.NoBrush)

        # draw horizontal lines
        for i in range(15):
            if i == 1 or i == 5 or i == 10 or i == 14:
                painter.drawLine(60, i*60, 5*60, i*60)
                painter.drawLine(10*60, i*60, 14*60, i*60)
            if i < 6 or i > 9:
                painter.drawLine(6*60, i*60, 9*60, i*60)
            if i == 6 or i == 9:
                painter.drawLine(0, i*60, 900, i*60)
            if i == 7 or i == 8:
                painter.drawLine(0, i*60, 6*60, i*60)
                painter.drawLine(9*60, i*60, 900, i*60)

        # draw vertical lines
        for i in range(15):
            if i == 1 or i == 5 or i == 10 or i == 14:
                painter.drawLine(i*60, 60, i*60, 5*60)
                painter.drawLine(i*60, 10*60, i*60, 14*60)
            if i < 6 or i > 9:
                painter.drawLine(i*60, 6*60, i*60, 9*60)
            if i == 6 or i == 9:
                painter.drawLine(i*60, 0, i*60, 900)
            if i == 7 or i == 8:
                painter.drawLine(i*60, 0, i*60, 6*60)
                painter.drawLine(i*60, 9*60, i*60, 900)

        # draw middle
        painter.drawLine(6*60, 6*60, 450, 450)
        painter.drawLine(6*60, 9*60, 450, 450)
        painter.drawLine(9*60, 6*60, 450, 450)
        painter.drawLine(9*60, 9*60, 450, 450)

        # draw circles
        # top left
        painter.setBrush(QtGui.QBrush(RED))
        painter.drawEllipse(1.5*60, 1.5*60, 60, 60)
        painter.drawEllipse(1.5*60, 3.5*60, 60, 60)
        painter.drawEllipse(3.5*60, 1.5*60, 60, 60)
        painter.drawEllipse(3.5*60, 3.5*60, 60, 60)
        # top right
        painter.setBrush(QtGui.QBrush(GREEN))
        painter.drawEllipse(10.5*60, 1.5*60, 60, 60)
        painter.drawEllipse(10.5*60, 3.5*60, 60, 60)
        painter.drawEllipse(12.5*60, 1.5*60, 60, 60)
        painter.drawEllipse(12.5*60, 3.5*60, 60, 60)
        # bottom left
        painter.setBrush(QtGui.QBrush(BLUE))
        painter.drawEllipse(1.5*60, 10.5*60, 60, 60)
        painter.drawEllipse(1.5*60, 12.5*60, 60, 60)
        painter.drawEllipse(3.5*60, 10.5*60, 60, 60)
        painter.drawEllipse(3.5*60, 12.5*60, 60, 60)
        # bottom right
        painter.setBrush(QtGui.QBrush(YELLOW))
        painter.drawEllipse(10.5*60, 10.5*60, 60, 60)
        painter.drawEllipse(10.5*60, 12.5*60, 60, 60)
        painter.drawEllipse(12.5*60, 10.5*60, 60, 60)
        painter.drawEllipse(12.5*60, 12.5*60, 60, 60)

        # draw arrows
        pen = QPen(QtCore.Qt.black)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.NoBrush)
        # left
        painter.drawLine(12, 450, 48, 450)
        painter.drawLine(48, 450, 35, 465)
        painter.drawLine(48, 450, 35, 435)
        # top
        painter.drawLine(450, 12, 450, 48)
        painter.drawLine(450, 48, 465, 35)
        painter.drawLine(450, 48, 435, 35)
        # right
        painter.drawLine(888, 450, 852, 450)
        painter.drawLine(852, 450, 865, 465)
        painter.drawLine(852, 450, 865, 435)
        # bottom
        painter.drawLine(450, 888, 450, 852)
        painter.drawLine(450, 852, 465, 865)
        painter.drawLine(450, 852, 435, 865)

    def draw_pieces(self, painter: QtGui.QPainter) -> None:
        pen = QPen(QtCore.Qt.black)
        pen.setWidth(1)
        painter.setPen(pen)

        board = self.app.state[0]

        # red
        painter.setBrush(RED_PIECE)
        # piece 1
        if np.sum(board[0]) == 1:
            idx = np.argmax(board[0])
            if idx == 0:
                painter.drawEllipse(97.5, 97.5, 45, 45)
            else:
                rectx, recty = RED_INDEX_TO_POS[idx]
                painter.drawEllipse(rectx + 7.5, recty + 7.5, 45, 45)
        # piece 2
        if np.sum(board[1]) == 1:
            idx = np.argmax(board[1])
            if idx == 0:
                painter.drawEllipse(217.5, 97.5, 45, 45)
            else:
                rectx, recty = RED_INDEX_TO_POS[idx]
                painter.drawEllipse(rectx + 7.5, recty + 7.5, 45, 45)
        # piece 3
        if np.sum(board[2]) == 1:
            idx = np.argmax(board[2])
            if idx == 0:
                painter.drawEllipse(97.5, 217.5, 45, 45)
            else:
                rectx, recty = RED_INDEX_TO_POS[idx]
                painter.drawEllipse(rectx + 7.5, recty + 7.5, 45, 45)
        # piece 4
        if np.sum(board[3]) == 1:
            idx = np.argmax(board[3])
            if idx == 0:
                painter.drawEllipse(217.5, 217.5, 45, 45)
            else:
                rectx, recty = RED_INDEX_TO_POS[idx]
                painter.drawEllipse(rectx + 7.5, recty + 7.5, 45, 45)

        # yellow
        painter.setBrush(YELLOW_PIECE)
        # piece 1
        if np.sum(board[4]) == 1:
            idx = np.argmax(board[4])
            if idx == 0:
                painter.drawEllipse(637.5, 637.5, 45, 45)
            else:
                rectx, recty = YELLOW_INDEX_TO_POS[idx]
                painter.drawEllipse(rectx + 7.5, recty + 7.5, 45, 45)
        # piece 2
        if np.sum(board[5]) == 1:
            idx = np.argmax(board[5])
            if idx == 0:
                painter.drawEllipse(757.5, 637.5, 45, 45)
            else:
                rectx, recty = YELLOW_INDEX_TO_POS[idx]
                painter.drawEllipse(rectx + 7.5, recty + 7.5, 45, 45)
        # piece 3
        if np.sum(board[6]) == 1:
            idx = np.argmax(board[6])
            if idx == 0:
                painter.drawEllipse(637.5, 757.5, 45, 45)
            else:
                rectx, recty = YELLOW_INDEX_TO_POS[idx]
                painter.drawEllipse(rectx + 7.5, recty + 7.5, 45, 45)
        # piece 4
        if np.sum(board[7]) == 1:
            idx = np.argmax(board[7])
            if idx == 0:
                painter.drawEllipse(757.5, 757.5, 45, 45)
            else:
                rectx, recty = YELLOW_INDEX_TO_POS[idx]
                painter.drawEllipse(rectx + 7.5, recty + 7.5, 45, 45)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        print(event.pos())

