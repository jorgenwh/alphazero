import os
import torch
import datetime

from .network import Network
from .config import Config

def save_checkpoint(dir_name: str, network: Network, games_played: int) -> None:
    filename = f"{dir_name}/model_checkpoint_{games_played}games.pt"
    if os.path.isfile(filename):
        return
    torch.save(network.model.state_dict(), filename)

def load_checkpoint(dir_name: str, network: Network, games_played: int) -> None:
    filename = f"{dir_name}/model_checkpoint_{games_played}games.pt"
    assert os.path.isfile(filename)
    network.model.load_state_dict(torch.load(filename))

def load_model(path: str, network: Network) -> None:
    assert os.path.isfile(path)
    network.model.load_state_dict(torch.load(path))

def setup_training_session(config: Config, game: str) -> str:
    if not os.path.isdir("output"):
        os.mkdir("output")

    inp = input("name this training session's directory: ")
    while os.path.isdir("output/" + inp):
        inp = input("directory already exists, please choose another name: ")
    dir_name = "output/" + inp + "/"
    os.mkdir(dir_name)

    f = open(dir_name + "config", "w")
    f.write("GAME=" + game + "\n")
    f.write("RESIDUAL_BLOCKS=" + str(config.RESIDUAL_BLOCKS) + "\n")
    f.close()

    return dir_name

def get_time_stamp(s):
    t_s = str(datetime.timedelta(seconds=round(s)))
    ts = t_s.split(':')
    return '(' + ts[0] + 'h ' + ts[1] + 'm ' + ts[2] + 's)'


class PrintColors():
    red = "\33[91m"
    green = "\33[92m"
    yellow = "\33[93m"
    blue = "\33[94m"
    bold = "\33[1m"
    transparent = "\33[90m"
    endc = "\33[0m"


class AverageMeter():
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f"{round(self.avg, 4)}"

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
