import torch
import numpy as np
from tqdm import tqdm

from .connect4_model import Connect4_Model
from .connect4_model import Connect4_ResNet, BasicBlock, Dual_Conv
from utils import Average_Meter

class Connect4_Network:
    def __init__(self, game_rules, args):
        self.game_rules = game_rules
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and self.args.cuda else "cpu")
        #self.model = Connect4_Model(self.args).to(self.device)
        #self.model = Connect4_ResNet(BasicBlock, [3, 4, 6, 3], num_classes=7).to(self.device)
        self.model = Dual_Conv(self.args)
        self.model.to_device(self.device)
        
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.args.lr)
        
    def evaluate(self, board):
        self.model.eval()
        b = board.reshape(1, 1, 6, 7)
        b = torch.FloatTensor(b).to(self.device)

        with torch.no_grad():
            pi, v = self.model(b)

        pi = torch.softmax(pi[0].cpu(), dim=0).data.numpy()
        v = v[0].cpu().data.detach().numpy()
        return pi, v
        
    def train(self, training_data):
        self.model.train()
        for epoch in range(self.args.epochs):
            print(f"Epoch: {epoch+1}/{self.args.epochs}")
            steps = int(len(training_data) / self.args.batch_size)
            epoch_loss = Average_Meter()

            t = tqdm(range(steps), desc="Training")
            for _ in t:
                indices = np.random.randint(len(training_data), size=self.args.batch_size)
                boards, pis, vs = list(zip(*[training_data[i] for i in indices]))
                
                boards = torch.FloatTensor(boards).to(self.device).reshape(self.args.batch_size, 1, 6, 7)
                pis = torch.FloatTensor(pis).to(self.device)
                vs = torch.FloatTensor(vs).to(self.device)

                out_pi, out_v = self.model(boards)

                pi_loss = self.CEL(pis, torch.log_softmax(out_pi, dim=1), pis.shape[0])
                v_loss = self.MSEL(vs, out_v, vs.shape[0])
                loss = pi_loss + v_loss

                epoch_loss.update(loss.item(), boards.shape[0])
                t.set_postfix(loss=epoch_loss)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
    def CEL(self, target, out, size):
        return -torch.sum(target * out) / size
    
    def MSEL(self, target, out, size):
        return torch.sum((target - out.reshape(-1)) ** 2) / size
