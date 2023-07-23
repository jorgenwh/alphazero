import numpy as np
from typing import Union

class Rules:
    def __init__(self):
        pass

    def get_start_state(self) -> Union[np.ndarray, tuple[np.ndarray, ...]]:
        raise NotImplementedError

    def get_action_space(self) -> int:
        raise NotImplementedError

    def get_valid_actions(self, 
            state: Union[np.ndarray, tuple[np.ndarray, ...]], 
            player: int
        ) -> np.ndarray:
        raise NotImplementedError

    def step(self, 
            state: Union[np.ndarray, tuple[np.ndarray, ...]],
            action: int,
            player: int
        ) -> Union[np.ndarray, tuple[np.ndarray, ...]]:
        raise NotImplementedError


    def flip_view(self, 
            state: Union[np.ndarray, tuple[np.ndarray, ...]]
        ) -> Union[np.ndarray, tuple[np.ndarray, ...]]:
        raise NotImplementedError

    def hash(self, 
            state: Union[np.ndarray, tuple[np.ndarray, ...]]
        ) -> int:
        raise NotImplementedError

    def get_winner(self, 
            state: Union[np.ndarray, tuple[np.ndarray, ...]]
        ) -> Union[int, None]:
        raise NotImplementedError

    def __str__(self) -> str:
        raise NotImplementedError
