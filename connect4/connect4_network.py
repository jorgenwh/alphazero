import torch
import numpy as np
from tqdm import tqdm

from .connect4_model import Connect4_Model
from utils import Average_Meter

class Connect4_Network:
    def __init__(self, game_rules, args):
        self.game_rules = game_rules
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and self.args.cuda else "cpu")
        self.model = Connect4_Model(self.args).to(self.device)
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.args.lr)
        
    def evaluate(self, board):
        """
        Evaluates a board position and gives a policy prediction aswell as a value prediction.
        The invalid actions are masked out of the policy using valid_actions before softmax or log_softmax is used.
        """
        self.model.eval()
        b = board.reshape(1, 1, 6, 7)
        b = torch.FloatTensor(b).to(self.device)

        with torch.no_grad():
            pi, v = self.model(b)

        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]
        
    def train(self, training_data):
        self.model.train()
        for epoch in range(self.args.epochs):
            print(f"Epoch: {epoch+1}/{self.args.epochs}")
            batches = int(len(training_data) / self.args.batch_size)

            pi_losses = Average_Meter()
            v_losses = Average_Meter()

            t = tqdm(range(batches), desc="Training")
            for _ in t:
                indices = np.random.randint(len(training_data), size=self.args.batch_size)
                boards, pis, vs = list(zip(*[training_data[i] for i in indices]))
                
                boards = torch.FloatTensor(boards).to(self.device)
                pis = torch.FloatTensor(pis).to(self.device)
                vs = torch.FloatTensor(vs).to(self.device)

                out_pi, out_v = self.model(boards)

                pi_loss = self.pi_loss(pis, out_pi)
                v_loss = self.v_loss(vs, out_v)
                loss = pi_loss + v_loss

                pi_losses.update(pi_loss.item(), boards.size(0))
                v_losses.update(v_loss.item(), boards.size(0))
                t.set_postfix(pi_loss=pi_losses, v_loss=v_losses)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
    def pi_loss(self, target, out):
        return -torch.sum(target * out) / target.size()[0]
    
    def v_loss(self, target, out):
        return torch.sum((target - out.view(-1)) ** 2) / target.size()[0]
