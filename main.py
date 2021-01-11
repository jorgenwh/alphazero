import os
import sys
from PyQt5 import QtGui, QtCore, QtWidgets

from alphazero import AlphaZero
from mcts import MCTS
from minimax import Minimax
from utils import load_model
from args import get_args

from games.connect4.connect4_rules import Connect4_Rules
from games.connect4.connect4_network import Connect4_Network
from games.connect4.connect4_window import Connect4_Window

from games.tictactoe.tictactoe_rules import TicTacToe_Rules
from games.tictactoe.tictactoe_network import TicTacToe_Network
from games.tictactoe.tictactoe_window import TicTacToe_Window

from games.gomoku.gomoku_rules import Gomoku_Rules
from games.gomoku.gomoku_network import Gomoku_Network
from games.gomoku.gomoku_window import Gomoku_Window

from games.othello.othello_rules import Othello_Rules
from games.othello.othello_network import Othello_Network
from games.othello.othello_window import Othello_Window

if __name__ == "__main__":
    args = get_args()
    games_sets = {
        "connect4": (Connect4_Rules, Connect4_Network, Connect4_Window),
        "tictactoe": (TicTacToe_Rules, TicTacToe_Network, TicTacToe_Window),
        "gomoku": (Gomoku_Rules, Gomoku_Network, Gomoku_Window),
        "othello": (Othello_Rules, Othello_Network, Othello_Window)
    }

    # Create the game rules object
    if args.game not in games_sets:
        raise NotImplementedError(f"Game '{args.game}' is not implemented!")

    if args.game == "gomoku":
        game_rules = games_sets[args.game][0](args.gomoku_size)
    elif args.game == "othello":
        assert args.othello_size % 2 == 0
        game_rules = games_sets[args.game][0](args.othello_size)
    else:
        game_rules = games_sets[args.game][0]()

    # Create a new neural network
    nnet = games_sets[args.game][1](game_rules, args)

    # If we are setting up a duel
    if args.duel:

        # If we are playing against a minimax agent
        if args.minimax:
            policy = Minimax(game_rules, args)

        # If we are playing against a MCTS with a neural net
        else:
            load_model(nnet, args.duel)
            policy = MCTS(game_rules, nnet, args)
        
        app = QtWidgets.QApplication(sys.argv)
        game_window = games_sets[args.game][2](game_rules, policy, args)
        sys.exit(app.exec_())
    
    # If we are training a neural net
    else:

        # If we are starting with a previous model
        if args.model:
            folder = "models/"
            if not os.path.isfile(os.path.join(folder, args.model)):
                print(f"Cannot find model '{os.path.join(folder, args.model)}'. Starting with a new model.")
            else:
                print(f"Loading pretrained model: '{os.path.join(folder, args.model)}'.")
                load_model(nnet, args.model)

        alphazero = AlphaZero(game_rules, nnet, args, games_sets[args.game][1])
        alphazero.train()
