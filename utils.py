import os
import torch

def setup_session():
    if not os.path.isdir("sessions/"):
        os.mkdir("sessions")

    num = 0
    while os.path.isdir("sessions/session" + str(num) + "/"):
        num += 1

    os.mkdir("sessions/session" + str(num))
    os.mkdir("sessions/session" + str(num) + "/model-checkpoints")
    return num

def save_checkpoint(nnet, sess_num, checkpoint_num):
    folder = "sessions/session" + str(sess_num) + "/model-checkpoints/"
    name = "nnet_checkpoint" + str(checkpoint_num)

    if os.path.isfile(os.path.join(folder, name)):
        raise FileExistsError(f"Model '{os.path.join(folder, name)}' already exists!")
    
    torch.save(nnet.model.state_dict(), os.path.join(folder, name))

def load_checkpoint(nnet, sess_num, checkpoint_num):
    folder = "sessions/session" + str(sess_num) + "/model-checkpoints/"
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
    Average meter from pytorch
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