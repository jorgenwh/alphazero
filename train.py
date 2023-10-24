import argparse
from alphazero import Trainer
from alphazero import Config, populate_config
from alphazero.misc import load_model

# The implemented games are:
# { Connect4, TicTacToe, Othello, Gomoku }
from alphazero import Connect4Rules, Connect4Network
from alphazero import TicTacToeRules, TicTacToeNetwork
from alphazero import OthelloRules, OthelloNetwork
from alphazero import GomokuRules, GomokuNetwork


if __name__ == "__main__":
    config = Config()

    # Parse command line arguments
    parser = argparse.ArgumentParser() 
    parser.add_argument("--game", choices=["Connect4", "TicTacToe", "Othello", "Gomoku"], required=True, help="The game to play. Accepted options are: { Connect4, TicTacToe, Othello, Gomoku }")
    parser.add_argument("--checkpoint", type=str, help="Path to a model checkpoint to load and begin training with.")
    parser.add_argument("--iterations", type=int, default=10, help="Number of training iterations to perform.")
    parser.add_argument("--episodes", type=int, default=80, help="Number of self-play games to play per iteration.")
    parser.add_argument("--replay_memory_size", type=int, default=50000, help="Size of the replay memory.")
    parser.add_argument("--evaluation_matches", type=int, default=40, help="Number of evaluation matches to perform at the end of each iteration.")
    parser.add_argument("--acceptance_threshold", type=float, default=0.55, help="Acceptance threshold for new checkpoints.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Determines the greediness of the agent. A high temperature means more deterministic and greedy play. Value must be in the (0.0, 1.0) range.")
    parser.add_argument("--monte_carlo_rollouts", type=int, default=120, help="Number of monte-carlo search rollouts to perform per action. More rollouts results in stronger play but longer think-time.")
    parser.add_argument("--cuda", type=bool, default=True, help="Whether or not to use CUDA.")
    parser.add_argument("--residual_blocks", type=int, default=2, help="Number of residual blocks in the neural network. Defaults to 2.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the neural network. Defaults to 0.001.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs for the neural network's weight optimizations. Defaults to 10.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size used when optimizing the neural network. Defaults to 128.")
    args = parser.parse_args()

    # Populate the config
    populate_config(
       config, 
       ITERATIONS=args.iterations, 
       EPISODES=args.episodes, 
       REPLAY_MEMORY_SIZE=args.replay_memory_size, 
       EVALUATION_MATCHES=args.evaluation_matches, 
       ACCEPTANCE_THRESHOLD=args.acceptance_threshold, 
       TEMPERATURE=args.temperature, 
       MONTE_CARLO_ROLLOUTS=args.monte_carlo_rollouts, 
       CUDA=args.cuda, 
       RESIDUAL_BLOCKS=args.residual_blocks, 
       LEARNING_RATE=args.learning_rate, 
       EPOCHS=args.epochs, 
       BATCH_SIZE=args.batch_size)

    # Initialize the game rules and neural network
    if args.game == "Connect4":
      network = Connect4Network(config)
      rules = Connect4Rules()
    if args.game == "TicTacToe":
      network = TicTacToeNetwork(config)
      rules = TicTacToeRules()
    if args.game == "Othello":
      network = OthelloNetwork(config)
      rules = OthelloRules()
    if args.game == "Gomoku":
      network = GomokuNetwork(config)
      rules = GomokuRules()

    # Load the pre-trained network if one is provided
    if args.checkpoint is not None:
        print(f"\nloading pre-trained network: {args.checkpoint}\n")
        load_model(args.checkpoint, network)

    # Initialize and start trainer
    trainer = Trainer(rules, network, config)
    trainer.start()
