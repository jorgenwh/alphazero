import numpy as np
from .config import Config
from .replay_memory import ReplayMemory

class Network():
    def __init__(self, config: Config):
        self.config = config

    def __call__(self, state: np.ndarray) -> tuple[np.ndarray, float]:
        raise NotImplementedError

    def train(self, replay_memory: ReplayMemory) -> None:
        raise NotImplementedError

    def batched_forward(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("batched_forward is not implemented for this network type")
