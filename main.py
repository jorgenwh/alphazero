import os
import sys
from PyQt5 import QtGui, QtCore, QtWidgets

from alphazero import AlphaZero
from mcts import MCTS
from minimax import Minimax
from utils import load_model
from args import args

from games.connect4.connect4_rules import Connect4Rules
from games.connect4.connect4_network import Connect4Network
from games.connect4.connect4_window import Connect4Window

from games.tictactoe.tictactoe_rules import TicTacToeRules
from games.tictactoe.tictactoe_network import TicTacToeNetwork
from games.tictactoe.tictactoe_window import TicTacToeWindow

from games.gomoku.gomoku_rules import GomokuRules
from games.gomoku.gomoku_network import GomokuNetwork
from games.gomoku.gomoku_window import GomokuWindow

from games.othello.othello_rules import OthelloRules
from games.othello.othello_network import OthelloNetwork
from games.othello.othello_window import OthelloWindow

from games.chess.chess_rules import ChessRules
from games.chess.chess_network import ChessNetwork
from games.chess.chess_window import ChessWindow

if __name__ == "__main__":
    sys.setrecursionlimit(2000)
    games_sets = {
        "connect4": (Connect4Rules, Connect4Network, Connect4Window),
        "tictactoe": (TicTacToeRules, TicTacToeNetwork, TicTacToeWindow),
        "gomoku": (GomokuRules, GomokuNetwork, GomokuWindow),
        "othello": (OthelloRules, OthelloNetwork, OthelloWindow),
        "chess": (ChessRules, ChessNetwork, ChessWindow)
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
    if args.duel or args.minimax:

        # If we are playing against a minimax agent (this option gets precedens over a nnet opponent)
        if args.minimax:
            policy = Minimax(game_rules, args)

        # If we are playing against a MCTS with a neural net
        else:
            load_model(nnet, "trained-models", args.duel)
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
                load_model(nnet, "models/", args.model)

        alphazero = AlphaZero(game_rules, nnet, args, games_sets[args.game][1])
        alphazero.train()
