

class MCTS:
    def __init__(self, game_env, nnet):
        self.game_env = game_env
        self.nnet = nnet