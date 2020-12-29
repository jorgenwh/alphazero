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
    
    params = nnet.model.state_dict()
    torch.save(params, os.path.join(folder, name))

def load_checkpoint(nnet, sess_num, checkpoint_num):
    folder = "sessions/session" + str(sess_num) + "/model-checkpoints/"
    name = "nnet_checkpoint" + str(checkpoint_num)
    assert os.path.isfile(os.path.join(folder, name))

    params = torch.load(os.path.join(folder, name))
    nnet.model.load_state_dict(params)

class Dotdict(dict):
    def __getattr__(self, name):
        return self[name]

class Average_Meter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f"{self.avg:.2e}"

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count