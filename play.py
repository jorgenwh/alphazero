import argparse
import sys
from PyQt5 import QtWidgets
from alphazero import Config, populate_config
from alphazero.misc import load_model

# The implemented games are:
# { Connect4, TicTacToe, Othello, Gomoku, Ludo }
from alphazero import Connect4Network, Connect4Rules, Connect4GUI
from alphazero import TicTacToeNetwork, TicTacToeRules, TicTacToeGUI
from alphazero import OthelloNetwork, OthelloRules, OthelloGUI
from alphazero import GomokuNetwork, GomokuRules, GomokuGUI


if __name__ == "__main__":
    config = Config()

    # Parse command line arguments
    parser = argparse.ArgumentParser() 
    parser.add_argument("--game", choices=["Connect4", "TicTacToe", "Othello", "Gomoku", "Ludo"], required=True, help="The game to play. Accepted options are: { Connect4, TicTacToe, Othello, Gomoku, Ludo }")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a model checkpoint to load and play against.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Determines the greediness of the agent. A high temperature means more deterministic and greedy play. Value must be in the (0.0, 1.0) range.")
    parser.add_argument("--monte_carlo_rollouts", type=int, default=160, help="Number of monte-carlo search rollouts to perform per action. More rollouts results in stronger play but longer think-time.")
    parser.add_argument("--residual_blocks", required=True, type=int, help="Number of residual blocks in the neural network. This must match the number of residual blocks used during training.")
    parser.add_argument("--nocuda", action="store_true", help="Disable CUDA.")
    args = parser.parse_args()

    # Populate the config
    populate_config(
      config, 
      TEMPERATURE=args.temperature, 
      MONTE_CARLO_ROLLOUTS=args.monte_carlo_rollouts, 
      CUDA=(not args.nocuda), 
      RESIDUAL_BLOCKS=args.residual_blocks, 
      LEARNING_RATE=0.001)

    # Initialize the game rules and neural network
    app = QtWidgets.QApplication(sys.argv)
    if args.game == "Connect4":
      network = Connect4Network(config)
      rules = Connect4Rules()
      gui = Connect4GUI(rules, network, config)
    if args.game == "TicTacToe":
      network = TicTacToeNetwork(config)
      rules = TicTacToeRules()
      gui = TicTacToeGUI(rules, network, config)
    if args.game == "Othello":
      network = OthelloNetwork(config)
      rules = OthelloRules()
      gui = OthelloGUI(rules, network, config)
    if args.game == "Gomoku":
      network = GomokuNetwork(config)
      rules = GomokuRules()
      gui = GomokuGUI(rules, network, config)

    # Load the pre-trained network checkpoint
    print(f"\nloading pre-trained network: {args.checkpoint}\n")
    load_model(args.checkpoint, network)

    sys.exit(app.exec_())
