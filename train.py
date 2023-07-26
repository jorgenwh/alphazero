import alphazero
from alphazero import Trainer
from alphazero import Connect4Rules as Rules
from alphazero import Connect4Network as Network

if __name__ == "__main__":
    network = Network()
    rules = Rules()
    trainer = Trainer(rules, network)
    trainer.start()
