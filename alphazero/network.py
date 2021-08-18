import numpy as np
from typing import Tuple, List

from alphazero.misc import Arguments

class Network():
    """
    Abstract network class.
    When implementing a new game, the details of the neural network used for that game might change.
    To implement a new neural network class structure, inherit from this class and implement
    all the below methods.
    """
    def __init__(self, args: Arguments):
        pass

    def __call__(self, board: np.ndarray) -> Tuple[np.ndarray, float]:
        raise NotImplementedError

    def train(self, training_examples: List[Tuple[np.ndarray, np.ndarray, float]]) -> None:
        raise NotImplementedError
