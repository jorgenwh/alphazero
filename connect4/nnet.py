import torch
import numpy as np
from tqdm import tqdm

from .connect4_network import Connect4_Model

class Connect4_Network:
    def __init__(self, game_rules, args):
        self.game_rules = game_rules
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and self.args.cuda else "cpu")
        self.model = Connect4_Model(self.args).to(self.device)
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.args.lr)

        self.v_loss_function = torch.nn.MSELoss()
        
    def evaluate(self, board, valid_actions):
        """
        Evaluates a board position and gives a policy prediction aswell as a value prediction.
        The invalid actions are masked out of the policy using valid_actions before softmax or log_softmax is used.
        """
        self.model.eval()
        b = board.reshape(1, 1, 6, 7)
        b = torch.Tensor(b).to(self.device)

        with torch.no_grad():
            pi, v = self.model(b)

        # mask out invalid actions from the policy tensor
        pi = pi[0].data.cpu()
        mask = torch.Tensor(valid_actions).type(torch.bool)

        actions = []
        for i in range(self.game_rules.get_action_space()):
            if valid_actions[i]:
                actions.append(i)

        masked_pi = torch.softmax(torch.masked_select(pi, mask), dim=0)
        pi = np.zeros(self.game_rules.get_action_space())

        for i, a in enumerate(actions):
            pi[a] = masked_pi[i]
        
        v = v[0].cpu().item()
        return pi, v
        
    def train(self, training_data):
        self.model.train()
        for epoch in range(self.args.epochs):
            print(f"Epoch: {epoch+1}/{self.args.epochs}")
            batches = int(len(training_data) / self.args.batch_size)

            for _ in tqdm(range(batches), desc="Update steps"):
                indices = np.random.randint(len(training_data), size=self.args.batch_size)
                board, pi, v = [], [], []
                for idx in indices:
                    board.append(training_data[idx][0])
                    pi.append(training_data[idx][1])
                    v.append(training_data[idx][2])
                
                board = torch.Tensor(board).to(self.device)
                pi = torch.Tensor(pi).to(self.device)
                v = torch.Tensor(v).to(self.device)

                out_pi, out_v = self.model(board)

                pi_loss = self.pi_loss(pi, out_pi)
                v_loss = self.v_loss_function(out_v.squeeze(), v)
                loss = pi_loss + v_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
    def pi_loss(self, target, out):
        return -torch.sum(target * out) / target.size()[0]
