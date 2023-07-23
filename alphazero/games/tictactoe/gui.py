import numpy as np
from PyQt5 import QtGui, QtCore, QtWidgets

from .rules import TicTacToeRules
from .network import TicTacToeNetwork
from ...mcts import MCTS
from ...misc import PrintColors as PC
from ...args import PLAY_TEMPERATURE

class TicTacToeGUI(QtWidgets.QMainWindow):
    def __init__(self, rules: TicTacToeRules, network: TicTacToeNetwork):
        super().__init__()
        self.rules = rules
        self.network = network

        self.mcts = MCTS(self.rules, self.network)

        # is the network playing as player 1 or -1
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
        self.setFixedSize(350, 350)
        self.centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.centralWidget)
        self.setGeometry(750, 280, 350, 350)

        self.tictactoe_widget = TicTacToeWidget(self.centralWidget, self)
        self.tictactoe_widget.setGeometry(40, 40, 270, 270)

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
            self.tictactoe_widget.draw()

    def player_step(self, action: int) -> None:
        if self.winner is not None:
            self.state = self.rules.get_start_state()
            self.winner = None
            self.cur_player = 1
            self.network_turn *= -1
            self.mcts = MCTS(self.rules, self.network)
            self.move = 1
            self.tictactoe_widget.draw()
        else:
            if self.rules.get_valid_actions(self.state, self.cur_player)[action] and self.winner is None:
                self.state = self.rules.step(self.state, action, self.cur_player)
                self.winner = self.rules.get_winner(self.state)
                self.cur_player *= -1
                self.tictactoe_widget.draw()

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
        print(f"{PC.transparent}|{PC.endc} {p1} (X)         win : {p1_color}{PC.bold}{lp1w}{PC.endc}% {PC.transparent}|{PC.endc}")
        print(f"{PC.transparent}|{PC.endc} {p2} (O)         win : {p2_color}{PC.bold}{lp2w}{PC.endc}% {PC.transparent}|{PC.endc}")
        print(f"{PC.transparent}|{PC.endc}                      Draw : {PC.bold}{ld}{PC.endc}% {PC.transparent}|{PC.endc}")
        print(f"{PC.transparent}| ----------------------------------- |{PC.endc}\n")

        self.move += 1

class TicTacToeWidget(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget, app: TicTacToeGUI):
        super().__init__(parent)
        self.app = app

        # list to contain the 3 slots making up the winning sequence.
        # this is used to color the winning sequence differently when the game is concluded
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
        for r in range(3):
            for c in range(3):
                if self.winner_row:
                    if (r, c) in self.winner_row:# or np.sum(self.app.state[:,r,c]) == 0:
                        painter.setOpacity(1)
                    else:
                        painter.setOpacity(0.5)
                else:
                    painter.setOpacity(1)

                if self.app.state[0,r,c] == 1:
                    self.draw_cross(painter, (r, c))
                elif self.app.state[1,r,c] == 1:
                    self.draw_circle(painter, (r, c))

    def draw_cross(self, painter: QtGui.QPainter, cell: tuple[int, int]) -> None:
        gap = self.frameGeometry().width()/3
        painter.drawLine(cell[1]*gap + gap/2 - 25, cell[0]*gap + gap/2 - 25, cell[1]*gap + gap/2 + 25, cell[0]*gap + gap/2 + 25)
        painter.drawLine(cell[1]*gap + gap/2 - 25, cell[0]*gap + gap/2 + 25, cell[1]*gap + gap/2 + 25, cell[0]*gap + gap/2 - 25)

    def draw_circle(self, painter: QtGui.QPainter, cell: tuple[int, int]) -> None:
        gap = self.frameGeometry().width()/3
        painter.drawEllipse(cell[1]*gap + 15, cell[0]*gap + 15, 60, 60)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        r = int(event.x() / 90)
        c = int(event.y() / 90)
        action = c*3 + r
        self.app.player_step(action)

    def get_winner_row(self) -> list[tuple[int, int]]:
        for j in range(2):
            for i in range(3):
                if self.app.state[j,i,0] == self.app.state[j,i,1] == self.app.state[j,i,2] != 0:
                    return [(i,0), (i,1), (i,2)]
                if self.app.state[j,0,i] == self.app.state[j,1,i] == self.app.state[j,2,i] != 0:
                    return [(0,i), (1,i), (2,i)]

            if self.app.state[j,0,0] == self.app.state[j,1,1] == self.app.state[j,2,2] != 0:
                return [(0,0), (1,1), (2,2)]
            if self.app.state[j,2,0] == self.app.state[j,1,1] == self.app.state[j,0,2] != 0:
                return [(2,0), (1,1), (0,2)]
        
        return []
