# network
START_MODEL: str = None
RESIDUAL_BLOCKS: int = 2
CUDA: bool = True

# algorithm
#ITERATIONS: int = 10 # USE THIS
ITERATIONS: int = 4
#EPISODES: int = 100 # USE THIS
EPISODES: int = 40
REPLAY_MEMORY_SIZE: int = 50000
EVALUATION_MATCHES: int = 40
ACCEPTANCE_THRESHOLD: float = 0.55
SELFPLAY_TEMPERATURE: float = 0.6
EVALUATION_TEMPERATURE: float = 1.0
PLAY_TEMPERATURE: float = 1.0
CPUCT: float = 1.0
MONTE_CARLO_LEAF_ROLLOUTS: int = 40

# network training
BATCH_SIZE: int = 64
LEARNING_RATE: float = 0.001
EPOCHS: int = 10

# other
GAME: str = "TicTacToe"