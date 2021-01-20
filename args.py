
class Arguments(dict):
    def __getattr__(self, attr):
        return self[attr]

args = Arguments({

    # AlphaZero
    "iterations": 100,
    "episodes": 120,
    "play_memory": 200_000,
    "eval_matches": 50,
    "acceptance_threshold": 0.55,
    "temperature": 1.0,
    "cpuct": 1.0,
    "monte_carlo_sims": 50,

    # Neural network
    "residual_blocks": 8,
    "learning_rate": 0.001,
    "epochs": 10,
    "batch_size": 128,
    "cuda": True,

    # Models
    "duel": None,
    "model": None,

    # Game and game-size
    "game": "chess",
    "minimax": "a",
    "gomoku_size": 19,
    "othello_size": 8
})
