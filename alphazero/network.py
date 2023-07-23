from collections import deque
from typing import Union
import numpy as np

class Network():
    def __init__(self):
        pass

    def __call__(self, 
            state: Union[np.ndarray, tuple[np.ndarray, ...]]
    ) -> tuple[np.ndarray, float]:
        raise NotImplementedError

    def train(self, replay_memory: deque) -> None:
        raise NotImplementedError
