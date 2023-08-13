from collections import deque
import torch
import numpy as np
from tqdm import tqdm

from ...replay_memory import ReplayMemory
from ...network import Network
from ...models import ResNet
from ...misc import AverageMeter
from ...args import CUDA, LEARNING_RATE, EPOCHS, BATCH_SIZE

def _CEL(target: torch.Tensor, out: torch.Tensor, size: int) -> torch.Tensor:
    return -torch.sum(target * out) / size

def _MSEL(target: torch.Tensor, out: torch.Tensor, size: int) -> torch.Tensor:
    return torch.sum((target - out.reshape(-1)) ** 2) / size


class OthelloNetwork(Network):
    def __init__(self):
        super().__init__()
        if CUDA:
            assert torch.cuda.is_available(), "CUDA is not available"
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        self.model = ResNet(in_height=8, in_width=8, in_channels=2, action_space=64).to(self.device)
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=LEARNING_RATE)

    def __call__(self, observation: np.ndarray) -> tuple[np.ndarray, float]:
        observation = observation.reshape(1, 2, 8, 8)
        observation = torch.from_numpy(observation).to(self.device)

        self.model.eval()
        with torch.no_grad():
            pi, v = self.model(observation)

        pi = torch.softmax(pi[0].cpu(), dim=0).data.numpy()
        v = v[0].cpu().data.detach().numpy()
        return pi, v

    def train(self, replay_memory: ReplayMemory) -> None:
        self.model.train()
        for epoch in range(EPOCHS):
            print(f"Epoch: {epoch+1}/{EPOCHS}")
            steps = int(len(replay_memory) / BATCH_SIZE)
            epoch_loss = AverageMeter()

            bar = tqdm(range(steps), desc="training", bar_format="{l_bar}{bar}| update: {n_fmt}/{total_fmt} - {unit} - elapsed: {elapsed}")
            for _ in bar:
                observations, pis, vs = replay_memory.get_random_batch(num_samples=BATCH_SIZE)

                observations = torch.from_numpy(observations).to(self.device)
                pis = torch.from_numpy(pis).to(self.device)
                vs = torch.from_numpy(vs).to(self.device)

                pi, v = self.model(observations)

                pi_loss = _CEL(pis, torch.log_softmax(pi, dim=1), pis.shape[0])
                v_loss = _MSEL(vs, v, vs.shape[0])
                loss = pi_loss + v_loss

                epoch_loss.update(loss.item(), observations.shape[0])
                bar.unit = f"loss: {epoch_loss}"

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

