import numpy as np

class ReplayMemory():
    def __init__(self, max_size: int, state_shape: int, action_space: int):
        self.max_size = max_size
        self.size = 0
        self.idx_cntr = 0
        self.states = np.zeros((max_size, *state_shape), dtype=np.float32)
        self.pis = np.zeros((max_size, action_space), dtype=np.float32)
        self.vs = np.zeros(max_size, dtype=np.float32)

    def __len__(self):
        return self.size

    def insert(self, state: np.ndarray, pi: np.ndarray, v: float) -> None:
        self.states[self.idx_cntr] = state
        self.pis[self.idx_cntr] = pi
        self.vs[self.idx_cntr] = v
        self.idx_cntr = (self.idx_cntr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def get_random_batch(self, num_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        indices = np.random.randint(0, self.size, num_samples)
        states = self.states[indices]
        pis = self.pis[indices]
        vs = self.vs[indices]
        return states, pis, vs
