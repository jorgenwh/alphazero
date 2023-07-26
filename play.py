import os
import sys
import torch
from PyQt5 import QtWidgets

# import whatever game you want to play 
# {Connect4, TicTacToe, Othello, Gomoku, Ludo}
from alphazero import LudoNetwork as Network
from alphazero import LudoRules as Rules
from alphazero import LudoGUI as GUI

if __name__ == "__main__":
    network = Network()
    rules = Rules()

    # load a pre-trained network
    name = None
    #name = "training/connect4/model_checkpoint_games.pt"

    if name is not None:
        print(f"loading pre-trained network: {name}")
        network.model.load_state_dict(torch.load(name))

    app = QtWidgets.QApplication(sys.argv)
    gui = GUI(rules, network)
    sys.exit(app.exec_())
