import os
import sys
import argparse
from PyQt5 import QtGui, QtCore, QtWidgets

from alphazero import AlphaZero
from mcts import MCTS
from utils import load_model

from games.connect4.connect4_rules import Connect4_Rules
from games.connect4.connect4_network import Connect4_Network
from games.connect4.connect4_window import Connect4_Window

from games.tictactoe.tictactoe_rules import TicTacToe_Rules
from games.tictactoe.tictactoe_network import TicTacToe_Network
from games.tictactoe.tictactoe_window import TicTacToe_Window

from games.gomoku.gomoku_rules import Gomoku_Rules
from games.gomoku.gomoku_network import Gomoku_Network
from games.gomoku.gomoku_window import Gomoku_Window

if __name__ == "__main__":
    games_sets = {
        "connect4": (Connect4_Rules, Connect4_Network, Connect4_Window),
        "tictactoe": (TicTacToe_Rules, TicTacToe_Network, TicTacToe_Window),
        "gomoku": (Gomoku_Rules, Gomoku_Network, Gomoku_Window)
    }

    parser = argparse.ArgumentParser(description="AZ.")

    # AlphaZero
    parser.add_argument("--iterations", help="Number of training iterations.", type=int, default=20)
    parser.add_argument("--episodes", help="Number of self-play episodes to perform per iteration.", type=int, default=150)
    parser.add_argument("--play_memory", help="Maximum number of example moves to remember from the self-plays.", type=int, default=150_000)
    parser.add_argument("--eval_matches", help="Number of evaluation playoffs against the previous neural network checkpoint.", type=int, default=50)
    parser.add_argument("--eval_score_threshold", help="Win/loss ratio threshold to save new neural networks.", type=float, default=0.55)
    parser.add_argument("--exploration_temp_threshold", help="How many moves to perform before decreasing the exploration threshold.", type=int, default=12)
    parser.add_argument("--cpuct", help="Constant to control the amount of exploration.", type=float, default=1.0)
    parser.add_argument("--monte_carlo_simulations", help="Number of monte carlo simulations to perform when choosing a move.", type=int, default=75)

    # Neural Network
    parser.add_argument("--lr", help="Learning rate for the neural network.", type=float, default=1e-3)
    parser.add_argument("--epochs", help="Training epochs per iteration.", type=int, default=10)
    parser.add_argument("--batch_size", help="How many move examples to sample for each batch during training. (2048 - 4096 in papers)", type=float, default=128)
    parser.add_argument("--cuda", help="Enable cuda.", type=bool, default=True)

    # Models
    parser.add_argument("--duel", help="The name of a model the user wants to play against. The training algorithm will not be ran if this is provided. Instead a game window will allow the user to play against AlphaZero using the model name provided.", type=str, default=None)
    parser.add_argument("--model", help="Start training with a pretrained model under 'models/[model]'.", type=str, default=None)

    # Which game to train/play
    parser.add_argument("--game", help="Which of the implemented game-rules and networks to provide to AlphaZero.", type=str, default="connect4")
    parser.add_argument("--size", help="Game board size. Only relevant for Gomoku and Go", type=int, default=7)

    args = parser.parse_args()

    if args.game not in games_sets:
        raise NotImplementedError(f"Game '{args.game}' is not implemented!")

    if args.game == "gomoku" or args.game == "go":
        assert args.size >= 6
        game_rules = games_sets[args.game][0](args.size)
    else:
        game_rules = games_sets[args.game][0]()
    nnet = games_sets[args.game][1](game_rules, args)

    if args.duel:
        load_model(nnet, args.duel)
        mcts = MCTS(game_rules, nnet, args)
        app = QtWidgets.QApplication(sys.argv)
        game_window = games_sets[args.game][2](game_rules, mcts, args)
        sys.exit(app.exec_())
    else:
        if args.model:
            folder = "models/"
            if not os.path.isfile(os.path.join(folder, args.model)):
                print(f"Cannot find model '{os.path.join(folder, args.model)}'. Starting with a new model.")
            else:
                print(f"Loading pretrained model: '{os.path.join(folder, args.model)}'.")
                load_model(nnet, args.model)

        alphazero = AlphaZero(game_rules, nnet, args, games_sets[args.game][1])
        alphazero.train()
