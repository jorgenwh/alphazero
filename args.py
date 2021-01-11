def get_args():
    import argparse
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
    parser.add_argument("--res_blocks", help="Number of residual blocks in residual neural network models.", type=int, default=8)
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
    return args
