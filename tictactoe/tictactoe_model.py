import torch
import torch.nn as nn

class TicTacToe_Model(nn.Module):
    def __init__(self, args):
        super(TicTacToe_Model, self).__init__()
        self.args = args
        self.board_height, self.board_width = 3, 3

        self.fc1 = self.linear_block(self.board_height * self.board_width, 512)
        self.fc2 = self.linear_block(512, 256)
        self.fc3 = self.linear_block(256, 128)

        self.pi = nn.Linear(128, 9)
        self.value = nn.Linear(128, 1)
        
    def forward(self, b):
        b = b.reshape(-1, self.board_height * self.board_width)
        r = self.fc1(b)
        r = self.fc2(r)
        r = self.fc3(r)
        
        pi = torch.log_softmax(self.pi(r), dim=1)
        value = torch.tanh(self.value(r))
        return pi, value

    def linear_block(self, in_features, out_features):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU()
        )