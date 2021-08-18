import numpy as np
from tqdm import tqdm
from typing import Tuple

from alphazero.rules import Rules
from alphazero.network import Network
from alphazero.mcts import MCTS
from alphazero.misc import Arguments

def play_episode(rules: Rules, mcts1: MCTS, mcts2: MCTS) -> float:
    """
    Play out a single match between two networks.

    Args:
        mcts1 (MCTS): the monte-carlo search tree that is providing the policy for player 1.
        mcts2 (MCTS): the monte-carlo search tree that is providing the policy for player 2.
    Returns:
        (float): the game result (as seen from player 1's perspective).
    """
    cur_player = 1
    board = rules.get_start_board()

    while not rules.is_concluded(board):
        perspective = board if cur_player == 1 else rules.flip(board)
        pi = mcts1.get_policy(perspective, temperature=0) if cur_player == 1 else mcts2.get_policy(perspective, temperature=0)
        action = np.argmax(pi)
        board = rules.step(board, action, cur_player)
        cur_player *= -1

    return rules.get_result(board)

def evaluate(rules: Rules, network: Network, checkpoint_network: Network, args: Arguments) -> Tuple[int, int, int]:
    """
    Evaluate a network against the current best checkpoint network by playing several matches.

    Args:
        rules (Rules): the game rules.
        network (Network): the neural network that is being evaluated.
        checkpoint_network (Network): the current best network checkpoint.
        args (Arguments): session arguments.
    Returns:
        (tuple): a tuple containing the following three datapoints:
            - the number of times the new network won against the checkpoint network
            - the number of times the new network tied with the checkpoint network
            - the number of times the new network lost against the checkpoint network
    """
    wins = 0
    losses = 0
    played = 0

    mcts1 = MCTS(rules, network, args)
    mcts2 = MCTS(rules, checkpoint_network, args)
    player = 1
    
    t = tqdm(range(args.eval_matches), desc="Evaluating")
    for _ in t:
        result = play_episode(rules, mcts1, mcts2) if player == 1 else play_episode(rules, mcts2, mcts1)
        played += 1
        wins += (player == 1 and result == 1) or (player == -1 and result == -1)
        losses += (player == 1 and result == -1) or (player == -1 and result == 1)
        t.set_postfix({"W/T/L": f"{wins}/{played - (wins + losses)}/{losses}"})

    return wins, played - (wins + losses), losses
    
   
