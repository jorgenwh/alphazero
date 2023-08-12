import torch
import alphazero
from alphazero import Trainer
from alphazero import Connect4Rules as Rules
from alphazero import Connect4Network as Network

if __name__ == "__main__":
    network = Network()
    rules = Rules()

    # load a pre-trained network
    name = None
    #name = ""

    if name is not None:
        print(f"loading pre-trained network: {name}")
        network.model.load_state_dict(torch.load(name))

    trainer = Trainer(rules, network)
    trainer.start()
