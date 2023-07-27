import alphazero
from alphazero import Trainer
from alphazero import LudoRules as Rules
from alphazero import LudoNetwork as Network

if __name__ == "__main__":
    network = Network()
    rules = Rules()
    trainer = Trainer(rules, network)
    trainer.start()
