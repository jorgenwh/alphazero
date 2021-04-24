import os
import torch
import datetime

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
    while os.path.isdir("sessions/" + args.game + "_session_" + str(num)):
        num += 1

    os.mkdir("sessions/" + args.game + "_session_" + str(num))
    os.mkdir("sessions/" + args.game + "_session_" + str(num) + "/model-checkpoints")

    f = open("sessions/" + args.game + "_session_" + str(num) + "/info.txt", "w")
    content = f"Session {num}\n\nGame: {game_rules.name()}"
    if hasattr(game_rules, "size"):
        content += f" (size: {game_rules.size})"
    content += f"\n\nStarted at: {datetime.datetime.now()}"[:-7] + f"\n\nResidual blocks: {args.residual_blocks}"
    f.write(content)

    return num

def save_checkpoint(nnet, sess_num, checkpoint_num, args):
    folder = "sessions/" + args.game + "_session_" + str(sess_num) + "/model-checkpoints/"
    name = "nnet_checkpoint" + str(checkpoint_num)

    if os.path.isfile(os.path.join(folder, name)):
        raise FileExistsError(f"Model '{os.path.join(folder, name)}' already exists!")
    
    torch.save(nnet.model.state_dict(), os.path.join(folder, name))

def load_checkpoint(nnet, sess_num, checkpoint_num, args):
    folder = "sessions/" + args.game + "_session_" + str(sess_num) + "/model-checkpoints/"
    name = "nnet_checkpoint" + str(checkpoint_num)
    assert os.path.isfile(os.path.join(folder, name))

    nnet.model.load_state_dict(torch.load(os.path.join(folder, name)))

def save_model(nnet, folder, name):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder '{folder}' not found.")

    n = 0
    while os.path.isfile(folder + name + str(n)):
        n += 1
    
    torch.save(nnet.model.state_dict(), os.path.join(folder, name + str(n)))

def load_model(nnet, folder, name):
    if not os.path.isfile(os.path.join(folder, name)):
        raise FileNotFoundError(f"Cannot find model '{os.path.join(folder, name)}'")

    nnet.model.load_state_dict(torch.load(os.path.join(folder, name)))

def get_time_stamp(s):
    t_s = str(datetime.timedelta(seconds=round(s)))
    ts = t_s.split(':')
    return '(' + ts[0] + 'h ' + ts[1] + 'm ' + ts[2] + 's)'

class AverageMeter(object):
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