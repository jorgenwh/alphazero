import os
import torch
from datetime import datetime

def setup_session(game_rules, args):
    """
    Setup a session folder containing the model-checkpoints folder.
    
    Returns:
        (int): the session number which is used to keep track of where to save the checkpoints
            and other data.
    """
    if not os.path.isdir("sessions/"):
        os.mkdir("sessions")

    num = 0
    while os.path.isdir("sessions/session_" + args.game + "_" + str(num)):
        num += 1

    os.mkdir("sessions/session_" + args.game + "_" + str(num))
    os.mkdir("sessions/session_" + args.game + "_" + str(num) + "/model-checkpoints")

    f = open("sessions/session_" + args.game + "_" + str(num) + "/info.txt", "w")
    content = f"Session {num}\n\nGame: {game_rules.name()}"
    if hasattr(game_rules, "size"):
        content += f" (size: {game_rules.size})"
    content += f"\n\nStarted at: {datetime.now()}"[:-7] + f" (Y-M-D H:M:S)\n\nResidual blocks: {args.res}"
    f.write(content)

    return num

def save_checkpoint(nnet, sess_num, checkpoint_num, args):
    folder = "sessions/session_" + args.game + "_" + str(sess_num) + "/model-checkpoints/"
    name = "nnet_checkpoint" + str(checkpoint_num)

    if os.path.isfile(os.path.join(folder, name)):
        raise FileExistsError(f"Model '{os.path.join(folder, name)}' already exists!")
    
    torch.save(nnet.model.state_dict(), os.path.join(folder, name))

def load_checkpoint(nnet, sess_num, checkpoint_num, args):
    folder = "sessions/session_" + args.game + "_" + str(sess_num) + "/model-checkpoints/"
    name = "nnet_checkpoint" + str(checkpoint_num)
    assert os.path.isfile(os.path.join(folder, name))

    nnet.model.load_state_dict(torch.load(os.path.join(folder, name)))

def save_model(nnet, name):
    if not os.path.isdir("models/"):
        os.mkdir("models")

    n = 0
    while os.path.isfile("models/" + name + str(n)):
        n += 1
    
    torch.save(nnet.model.state_dict(), os.path.join("models/", name + str(n)))

def load_model(nnet, name):
    folder = "models/"
    if not os.path.isfile(os.path.join(folder, name)):
        raise FileNotFoundError(f"Cannot find model '{os.path.join(folder, name)}'")

    nnet.model.load_state_dict(torch.load(os.path.join(folder, name)))

class Average_Meter(object):
    """
    Average meter from pytorch.
    """
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f"{round(self.avg, 3)}"

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        