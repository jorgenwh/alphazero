import torch
import numpy as np
from tqdm import tqdm

from .tictactoe_model import TicTacToe_Model
from utils import Average_Meter

class TicTacToe_Network:
    def __init__(self, game_rules, args):
        self.game_rules = game_rules
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and self.args.cuda else "cpu")
        self.model = TicTacToe_Model(self.args).to(self.device)
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.args.lr)
        
    def evaluate(self, board):
        self.model.eval()
        b = board.reshape(1, 1, 3, 3)
        b = torch.FloatTensor(b).to(self.device)

        with torch.no_grad():
            pi, v = self.model(b)

        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]
        
    def train(self, training_data):
        self.model.train()
        for epoch in range(self.args.epochs):
            print(f"Epoch: {epoch+1}/{self.args.epochs}")
            steps = int(len(training_data) / self.args.batch_size)

            pi_l = Average_Meter()
            v_l = Average_Meter()

            t = tqdm(range(steps), desc="Training")
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

                pi_l.update(pi_loss.item(), boards.size(0))
                v_l.update(v_loss.item(), boards.size(0))
                t.set_postfix(pi_loss=pi_l, v_loss=v_l)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
    def pi_loss(self, target, out):
        return -torch.sum(target * out) / target.size()[0]
    
    def v_loss(self, target, out):
        return torch.sum((target - out.view(-1)) ** 2) / target.size()[0]
