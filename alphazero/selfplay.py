import numpy as np
from tqdm import tqdm
from typing import List, Tuple

from alphazero.rules import Rules
from alphazero.network import Network
from alphazero.mcts import MCTS
from alphazero.misc import Arguments

def play_episode(mcts: MCTS, rules: Rules, args: Arguments) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    sequence = []
    board = rules.get_start_board()
    cur_player = 1
    ply = 0

    while not rules.is_concluded(board):
        perspective = board if cur_player == 1 else rules.flip(board)
        pi = mcts.get_policy(perspective, args.temperature)

        sequence.append((perspective, pi, cur_player))

        action = np.random.choice(rules.get_action_space(), p=pi)
        board = rules.step(board, action, cur_player)
        cur_player *= -1

    value = rules.get_result(board)
    for i, (board, pi, player) in enumerate(sequence):
        v = (1 if player == value else -1) if value else 0
        sequence[i] = (board, pi, v)

    return sequence

def selfplay(rules: Rules, network: Network, args: Arguments) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    training_examples = []
    
    for iter in tqdm(range(args.episodes), desc="Self-Play", bar_format="{l_bar}{bar}| Game: {n_fmt}/{total_fmt} - Elapsed: {elapsed}"):
        mcts = MCTS(rules, network, args)
        training_examples.extend(play_episode(mcts, rules, args))

    return training_examples
