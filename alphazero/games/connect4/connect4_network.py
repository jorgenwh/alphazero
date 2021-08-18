import torch
import numpy as np
from tqdm import tqdm
from typing import List, Tuple

from .connect4_model import Connect4Model
from alphazero.network import Network
from alphazero.misc import Arguments, AverageMeter

class Connect4Network(Network):
    def __init__(self, args: Arguments):
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and self.args.cuda else "cpu")
        self.model = Connect4Model(self.args).to(self.device)
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.args.learning_rate)
        
    def __call__(self, board: np.ndarray) -> Tuple[np.ndarray, float]:
        self.model.eval()
        b = board.reshape(1, 1, 6, 7)
        b = torch.FloatTensor(b).to(self.device)

        with torch.no_grad():
            pi, v = self.model(b)

        pi = torch.softmax(pi[0].cpu(), dim=0).data.numpy()
        v = v[0].cpu().data.detach().numpy()
        return pi, v
        
    def train(self, training_examples: List[Tuple[np.ndarray, np.ndarray, float]]) -> None:
        self.model.train()
        for epoch in range(self.args.epochs):
            print(f"Epoch: {epoch+1}/{self.args.epochs}")
            steps = int(len(training_examples) / self.args.batch_size)
            epoch_loss = AverageMeter()

            bar = tqdm(range(steps), desc="Training", bar_format="{l_bar}{bar}| Updated: {n_fmt}/{total_fmt} - {unit} - Elapsed: {elapsed}")
            for _ in bar:
                indices = np.random.randint(len(training_examples), size=self.args.batch_size)
                boards, pis, vs = list(zip(*[training_examples[i] for i in indices]))
                
                boards = torch.FloatTensor(boards).to(self.device).reshape(self.args.batch_size, 1, 6, 7)
                pis = torch.FloatTensor(pis).to(self.device)
                vs = torch.FloatTensor(vs).to(self.device)

                out_pi, out_v = self.model(boards)

                pi_loss = self.CEL(pis, torch.log_softmax(out_pi, dim=1), pis.shape[0])
                v_loss = self.MSEL(vs, out_v, vs.shape[0])
                loss = pi_loss + v_loss

                epoch_loss.update(loss.item(), boards.shape[0])
                bar.unit = f"Loss: {epoch_loss}"

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
    def CEL(self, target: torch.Tensor, out: torch.Tensor, size: int) -> torch.Tensor:
        return -torch.sum(target * out) / size
    
    def MSEL(self, target: torch.Tensor, out: torch.Tensor, size: int) -> torch.Tensor:
        return torch.sum((target - out.reshape(-1)) ** 2) / size
