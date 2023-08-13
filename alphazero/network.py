from collections import deque
from typing import Union
import numpy as np

from .replay_memory import ReplayMemory

class Network():
    def __init__(self):
        pass

    def __call__(self, 
            state: Union[np.ndarray, tuple[np.ndarray, ...]]
    ) -> tuple[np.ndarray, float]:
        raise NotImplementedError

    def train(self, replay_memory: ReplayMemory) -> None:
        raise NotImplementedError

    def batched_forward(self, 
            states: list[Union[np.ndarray, tuple[np.ndarray, ...]]]
    ) -> list[tuple[np.ndarray, float]]:
        raise NotImplementedError
