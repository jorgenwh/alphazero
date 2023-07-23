import numpy as np
from tqdm import tqdm
from collections import deque

from .rules import Rules
from .network import Network
from .mcts import MCTS
from .args import EPISODES, SELFPLAY_TEMPERATURE

def selfplay(rules: Rules, network: Network, replay_memory: deque) -> None:
    for i in tqdm(
            range(EPISODES), 
            desc="self-play", 
            bar_format="{l_bar}{bar}| game: {n_fmt}/{total_fmt} - elapsed: {elapsed}"
    ):
        mcts = MCTS(rules, network)
        play_episode(rules, mcts, replay_memory)
    return EPISODES

def play_episode(rules: Rules, mcts: MCTS, replay_memory: deque) -> None:
    state = rules.get_start_state() 
    sequence = deque()
    cur_player = 1
    winner = None

    while winner is None:
        observartion = state if cur_player == 1 else rules.flip_view(state)
        pi = mcts.get_policy(observartion, SELFPLAY_TEMPERATURE)
        sequence.append((observartion, pi, cur_player))
        action = np.random.choice(rules.get_action_space(), p=pi)
        state = rules.step(state, action, cur_player)
        winner = rules.get_winner(state)
        cur_player *= -1

    for i, (observation, pi, player) in enumerate(sequence):
        v = (1 if player == winner else -1) if winner else 0
        replay_memory.append((observation, pi, v))
