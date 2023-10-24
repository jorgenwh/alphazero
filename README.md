# A simple AlphaZero implementation in Python

<p align="center">
  <img src="./assets/ai-game-player.png" width="300">
</p>

## Usage

### Training a new neural network

To train a new neural network to play one of the implemented games, use the following command:

```python
python train.py --game <game_name>
```
Replace `<game_name>` with the name of the game you want to train the network for.

### Playing against a neural network

To play against a trained neural network using the QT GUI module, use the following command:

```python
python play.py --game <game_name> --network <network_path>
```
Replace `<game_name>` with the name of the game you want to play, and `<network_path>` with the path to the trained neural network.

