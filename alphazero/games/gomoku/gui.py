import numpy as np
from PyQt5 import QtGui, QtCore, QtWidgets

from .rules import GomokuRules, GOMOKU_BOARD_SIZE
from .network import GomokuNetwork
from ...mcts import MCTS
from ...misc import PrintColors as PC
from ...args import PLAY_TEMPERATURE

class GomokuGUI(QtWidgets.QMainWindow):
    def __init__(self, rules: GomokuRules, network: GomokuNetwork):
        super().__init__()
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), QtGui.QColor(255, 212, 129))
        self.setPalette(p)

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
        self.setFixedSize(GOMOKU_BOARD_SIZE*40 + 75, GOMOKU_BOARD_SIZE*40 + 75)
        self.centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.centralWidget)
        self.setGeometry(1500, 330, 350, 350)
        self.gomoku_widget = GomokuWidget(self.centralWidget, self)
        self.gomoku_widget.setGeometry(40, 40, GOMOKU_BOARD_SIZE*40, GOMOKU_BOARD_SIZE*40)

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
            self.gomoku_widget.draw()

    def player_step(self, action: int) -> None:
        if self.winner is not None:
            self.state = self.rules.get_start_state()
            self.winner = None
            self.cur_player = 1
            self.network_turn *= -1
            self.mcts = MCTS(self.rules, self.network)
            self.move = 1
            self.gomoku_widget.draw()
        else:
            if self.rules.get_valid_actions(self.state, self.cur_player)[action] and self.winner is None:
                self.state = self.rules.step(self.state, action, self.cur_player)
                self.winner = self.rules.get_winner(self.state)
                self.cur_player *= -1
                self.gomoku_widget.draw()

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
        print(f"{PC.transparent}|{PC.endc} {p2} (white)     win : {p2_color}{PC.bold}{lp2w}{PC.endc}% {PC.transparent}|{PC.endc}")
        print(f"{PC.transparent}|{PC.endc}                      Draw : {PC.bold}{ld}{PC.endc}% {PC.transparent}|{PC.endc}")
        print(f"{PC.transparent}| ----------------------------------- |{PC.endc}\n")

        self.move += 1

class GomokuWidget(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget, app: GomokuGUI):
        super().__init__(parent)
        self.app = app
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
        gap = right / GOMOKU_BOARD_SIZE

        for i in range(GOMOKU_BOARD_SIZE):
            painter.drawLine(gap/2 + i*gap, 0, gap/2 + i*gap, right)
            painter.drawLine(0, gap/2 + i*gap, bottom, gap/2 + i*gap)

    def draw_stones(self, painter: QtGui.QPainter) -> None:
        for r in range(GOMOKU_BOARD_SIZE):
            for c in range(GOMOKU_BOARD_SIZE):
                if self.winner_row:
                    if self.app.state[0,r,c] == 1:
                        self.draw_black(painter, (r, c), 1.0 if (r,c) in self.winner_row else 0.5)
                    elif self.app.state[1,r,c] == 1:
                        self.draw_white(painter, (r, c), 1.0 if (r,c) in self.winner_row else 0.5)
                else:
                    if self.app.state[0,r,c] == 1:
                        self.draw_black(painter, (r, c), 1.0)
                    elif self.app.state[1,r,c] == 1:
                        self.draw_white(painter, (r, c), 1.0)

    def draw_black(self, painter: QtGui.QPainter, intersection: tuple[int, int], opacity: float) -> None:
        painter.setBrush(QtGui.QBrush(QtGui.QColor(40, 40, 40)))
        painter.setOpacity(opacity)
        gap = self.frameGeometry().width() / GOMOKU_BOARD_SIZE
        x = 5 + intersection[1] * gap
        y = 5 + intersection[0] * gap
        painter.drawEllipse(x, y, gap*0.75, gap*0.75)
    
    def draw_white(self, painter: QtGui.QPainter, intersection: tuple[int, int], opacity: float) -> None:
        painter.setBrush(QtGui.QBrush(QtGui.QColor(215, 215, 215))) 
        painter.setOpacity(opacity)
        gap = self.frameGeometry().width() / GOMOKU_BOARD_SIZE
        x = 5 + intersection[1] * gap
        y = 5 + intersection[0] * gap
        painter.drawEllipse(x, y, gap*0.75, gap*0.75)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        x = int(event.x() / 40)
        y = int(event.y() / 40)
        action = y*GOMOKU_BOARD_SIZE + x
        self.app.player_step(action)

    def get_winner_row(self) -> list[tuple[int, int]]:
        for c in range(GOMOKU_BOARD_SIZE):
            for r in range(GOMOKU_BOARD_SIZE):
                if c < GOMOKU_BOARD_SIZE - 4:
                    if np.sum(self.app.state[0,r,c:c+5]) == 5 or np.sum(self.app.state[1,r,c:c+5]) == 5:
                        return [(r, c), (r,c+1), (r,c+2), (r,c+3), (r,c+4)]

                if r < GOMOKU_BOARD_SIZE - 4:
                    if np.sum(self.app.state[0,r:r+5,c]) == 5 or np.sum(self.app.state[1,r:r+5,c]) == 5:
                        return [(r, c), (r+1,c), (r+2,c), (r+3,c), (r+4,c)]

                if c < GOMOKU_BOARD_SIZE - 4 and r < GOMOKU_BOARD_SIZE - 4:
                    if self.app.state[0,r,c] == self.app.state[0,r+1,c+1] == self.app.state[0,r+2,c+2] == self.app.state[0,r+3,c+3] == self.app.state[0,r+4,c+4] == 1:
                        return [(r, c), (r+1,c+1), (r+2,c+2), (r+3,c+3), (r+4,c+4)]
                    if self.app.state[1,r,c] == self.app.state[1,r+1,c+1] == self.app.state[1,r+2,c+2] == self.app.state[1,r+3,c+3] == self.app.state[1,r+4,c+4] == 1:
                        return [(r, c), (r+1,c+1), (r+2,c+2), (r+3,c+3), (r+4,c+4)]
                if c < GOMOKU_BOARD_SIZE - 4 and r >= 4:
                    if self.app.state[0,r,c] == self.app.state[0,r-1,c+1] == self.app.state[0,r-2,c+2] == self.app.state[0,r-3,c+3] == self.app.state[0,r-4,c+4] == 1:
                        return [(r, c), (r-1,c+1), (r-2,c+2), (r-3,c+3), (r-4,c+4)]
                    if self.app.state[1,r,c] == self.app.state[1,r-1,c+1] == self.app.state[1,r-2,c+2] == self.app.state[1,r-3,c+3] == self.app.state[1,r-4,c+4] == 1:
                        return [(r, c), (r-1,c+1), (r-2,c+2), (r-3,c+3), (r-4,c+4)]
        return []
