# AlphaZero

<p align="center">
  <img src="./assets/ai-game-player.png" width="300">
</p>

This is a straightforward implementation of DeepMind's self-play reinforcement learning algorithm, AlphaZero. 
The code is written in Python, utilizing PyTorch for the deep neural networks and PyQt5 for the GUI applications, allowing users to play against the trained networks.

Given that self-play reinforcement learning demands significant computational resources and this implementation is designed for single-computer, single-GPU setups, it's challenging to train robust networks for games with a very large number of possible states, like Chess and Go. 
Nonetheless, this implementation supports simpler games. 
Currently, there are rules, networks, and GUIs for four games: TicTacToe, Connect4, Othello, and a condensed version of Gomoku.

## Installation and setup
To install the necessary dependencies:
```bash
pip install torch numpy tqdm pyqt5
```

Optionally, if you wish to install alphazero (although this is not necessary to use it):
```bash
pip install .
```

## Usage

### Training a new neural network

To train a new neural network to play one of the implemented games, use the following command:

```bash
python train.py --game <game_name>
```
Replace `<game_name>` with the name of the game you want to train the network for.
The currently implemented games are: { TicTacToe, Connect4, Othello, Gomoku }.

The following optional arguments can also be specified:

* `--checkpoint` (str): the path to a model checkpoint to begin training from.
* `--iterations` (int): the number of training iterations to perform. (default=10)
* `--episodes` (int): the number of self-play games to play per training iteration. (default=80)
* `--replay_memory_size` (int): the memory capacity of the replay memory. (default=50,000)
* `--evaluation_matches` (int): the number of evaluation matches to perform before deciding whether to accept a new neural network checkpoint. (default=40)
* `--acceptance_threshold` (float): the required ratio of matches won against the previous checkpoint to accept a new model. (default=0.55)
* `--temperature` (float): determines the greediness of the agent. A higher temperature means more greedy and deterministic play. The temperature must be in the range (0.0, 1.0). (default=1.0)
* `--monte_carlo_rollouts` (int): the number of monte-carlo rollouts to perform per action. A higher number of rollouts means stronger performance but increases think-time. (default=120)
* `--cuda` (bool): Whether or not to use CUDA. (default=True)
* `--residual_blocks` (int): number of residual blocks in the neural network. (default=2)
* `--learning_rate` (float): the learning rate for the neural network. (default=0.001)
* `--epochs` (int): the number of training epochs for the neural network. (default=10)
* `--batch_size` (int): the batch size used when training the neural network. (default=128)

### Playing against a neural network

To play against a trained neural network using PyQt5 GUI, use the following command:

```bash
python play.py --game <game_name> --checkpoint <checkpoint_path> --residual_blocks <num_residual_blocks>
```
Replace `<game_name>` with the name of the game you want to play, `<checkpoint_path>` with the path to a neural network checkpoint and `<num_residual_blocks>` with the number of residual blocks used in the checkpoint network.

The following optional arguments can also be specified:

* `--temperature` (float): determines the greediness of the agent. A higher temperature means more greedy and deterministic play. The temperature must be in the range (0.0, 1.0). (default=1.0)
* `--monte_carlo_rollouts` (int): the number of monte-carlo rollouts to perform per action. A higher number of rollouts means stronger performance but increases think-time. (default=120)
* `--cuda` (bool): Whether or not to use CUDA. (default=True)