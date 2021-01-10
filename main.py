import os
import sys
import argparse
from PyQt5 import QtGui, QtCore, QtWidgets

from alphazero import AlphaZero
from mcts import MCTS
from minimax import Minimax
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

from games.othello.othello_rules import Othello_Rules
from games.othello.othello_network import Othello_Network
from games.othello.othello_window import Othello_Window

if __name__ == "__main__":
    games_sets = {
        "connect4": (Connect4_Rules, Connect4_Network, Connect4_Window),
        "tictactoe": (TicTacToe_Rules, TicTacToe_Network, TicTacToe_Window),
        "gomoku": (Gomoku_Rules, Gomoku_Network, Gomoku_Window),
        "othello": (Othello_Rules, Othello_Network, Othello_Window)
    }

    parser = argparse.ArgumentParser(description="AZ.")

    # AlphaZero
    parser.add_argument("--iterations", help="Number of training iterations.", type=int, default=20)
    parser.add_argument("--episodes", help="Number of self-play episodes to perform per iteration.", type=int, default=115)
    parser.add_argument("--play_memory", help="Maximum number of training example moves to remember from self-play.", type=int, default=200_000)
    parser.add_argument("--eval_matches", help="Number of pit matches to evaluate the proportional performance of a neural network against another.", type=int, default=40)
    parser.add_argument("--eval_score_threshold", help="Win/loss ratio threshold to accept updated neural networks.", type=float, default=0.55)
    parser.add_argument("--temperature", help="Temperature controlling exploration (when choosing moves) during self-play.", type=float, default=1)
    parser.add_argument("--cpuct", help="Constant controlling exploration in MCTS.", type=float, default=1.0)
    parser.add_argument("--monte_carlo_sims", help="Number of monte-carlo simulations performed for each move chosen.", type=int, default=50)

    # Neural Network
    parser.add_argument("--lr", help="Neural network learning rate.", type=float, default=1e-3)
    parser.add_argument("--epochs", help="Neural network training epochs (per iteration).", type=int, default=10)
    parser.add_argument("--batch_size", help="Neural network training batch size.", type=int, default=64)
    parser.add_argument("--cuda", help="Enable cuda.", type=bool, default=True)

    # Models
    parser.add_argument("--duel", help="The name of a model the user wants to play against. The training algorithm will not be ran if this is provided. Instead, a game window will allow the user to play against AlphaZero using the nnet model provided.", type=str, default=None)
    parser.add_argument("--model", help="Start training with a pretrained model under 'models/[model]'.", type=str, default=None)

    # Which game to train/play
    parser.add_argument("--game", help="Which of the implemented game-rules to assume during the session.", type=str, default="connect4")
    parser.add_argument('--minimax', help="Use minimax in duel (and how deep it should search).", type=int, default=None)

    parser.add_argument("--gomoku_size", help="Game board size for Gomoku.", type=int, default=19)
    parser.add_argument("--othello_size", help="Game board size for Othello.", type=int, default=8)

    args = parser.parse_args()

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

    # EVALUATION TEST (TODO ELO SYSTEM)
    """nnet1 = games_sets[args.game][1](game_rules, args)
    nnet2 = games_sets[args.game][1](game_rules, args)

    load_model(nnet1, "later_model")
    load_model(nnet2, "earlier_model")

    from pit import Pit
    print(f"\nPit Evaluation")
    pit = Pit(game_rules, nnet1, nnet2, args)
    wins, ties, losses = pit.evaluate()
    score = wins / max((wins + losses), 1)
    print(f"W: {wins} - T: {ties} - L: {losses} - Score: {round(score, 3)}")
    exit()"""
    # EVALUATION TEST END

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
