import numpy as np
from tqdm import tqdm

from .rules import Rules
from .network import Network
from .mcts import MCTS
from .config import Config
from .misc import PrintColors as PC

def evaluate(
        rules: Rules,
        network: Network, 
        checkpoint_network: Network,
        config: Config) -> tuple[int, int, int]:
    wins, losses, played = 0, 0, 0
    mcts1 = MCTS(rules, network, config)
    mcts2 = MCTS(rules, checkpoint_network, config)
    player = 0

    bar = tqdm(range(config.EVALUATION_MATCHES), desc="evaluating", bar_format="{l_bar}{bar}| game: {n_fmt}/{total_fmt} - {unit} - elapsed: {elapsed}")
    for _ in bar:
        result = play_match(rules, mcts1, mcts2) if player == 0 else play_match(rules, mcts2, mcts1)
        player = (player + 1) % 2
        played += 1
        wins += 1 if result == 1 else 0
        losses += 1 if result == -1 else 0
        bar.unit = f"W/T/L - {PC.green}{wins}{PC.endc}/{played - (wins + losses)}/{PC.red}{losses}{PC.endc}"

    ties = played - (wins + losses)
    return wins, ties, losses

def play_match(rules: Rules, mcts1: MCTS, mcts2: MCTS) -> int:
    state = rules.get_start_state()
    cur_player = 1
    winner = None

    while winner is None:
        observation = state if cur_player == 1 else rules.flip_view(state)
        pi = mcts1.get_policy(observation) if cur_player == 1 else mcts2.get_policy(observation)
        action = np.random.choice(rules.get_action_space(), p=pi)
        state = rules.step(state, action, cur_player)
        winner = rules.get_winner(state)
        cur_player *= -1

    return winner
