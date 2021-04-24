class Rules:
    """
    Abstract game-rules class.
    For any new game, inherit from this class and implement all of the below methods
    """
    def __init__(self):
        pass

    def step(self, board, action, player):
        raise NotImplementedError

    def get_action_space(self):
        raise NotImplementedError

    def get_valid_actions(self, board, player):
        raise NotImplementedError

    def get_start_board(self):
        raise NotImplementedError

    def perspective(self, board, player):
        raise NotImplementedError

    def tostring(self, board):
        raise NotImplementedError

    def terminal(self, board):
        raise NotImplementedError

    def result(self, board, player):
        raise NotImplementedError

    def is_winner(self, board, player):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError