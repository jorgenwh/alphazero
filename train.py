import alphazero
from alphazero import Trainer
from alphazero import TicTacToeRules as Rules
from alphazero import TicTacToeNetwork as Network

if __name__ == "__main__":
    #alphazero.CUDA = False
    network = Network()
    rules = Rules()
    trainer = Trainer(rules, network)
    trainer.start()
