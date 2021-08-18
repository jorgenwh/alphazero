import torch
from typing import Tuple

from alphazero.misc import Arguments

class ResidualBlock(torch.nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(256)
        self.conv2 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(256)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        r = self.conv1(x)
        r = self.bn1(r)
        r = torch.relu(r)
        r = self.conv2(r)
        r = self.bn2(r)
        r += residual
        r = torch.relu(r)
        return r

class Connect4Model(torch.nn.Module):
    def __init__(self, args: Arguments):
        super(Connect4Model, self).__init__()
        self.args = args
        
        self.conv_block = torch.nn.Sequential(
            torch.nn.Conv2d(1, 256, kernel_size=(3, 3), stride=1, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU()
        )
        
        self.residual_blocks = []
        for _ in range(self.args.residual_blocks):
            self.residual_blocks.append(ResidualBlock())
        self.residual_tower = torch.nn.Sequential(*self.residual_blocks)
        
        # policy
        self.pi_conv_bn = self.conv_bn(256, 2, kernel_size=1, stride=1)
        self.pi = torch.nn.Linear(2 * 6 * 7, 7)

        # value
        self.v_conv_bn = self.conv_bn(256, 1, kernel_size=1, stride=1)
        self.v_fc = torch.nn.Linear(6 * 7, 256)
        self.v = torch.nn.Linear(256, 1)

    def conv_bn(self, in_channels: int, out_channels: int, *args, **kwargs) -> torch.nn.Sequential:
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, *args, **kwargs),
            torch.nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.reshape(x.shape[0], 1, 6, 7)
        r = self.conv_block(x)
        r = self.residual_tower(r)

        # policy
        pi = self.pi_conv_bn(r)
        pi = torch.relu(pi)
        pi = pi.reshape(pi.shape[0], -1)
        pi = self.pi(pi)

        # value
        v = self.v_conv_bn(r)
        v = torch.relu(v)
        v = v.reshape(v.shape[0], -1)
        v = self.v_fc(v)
        v = torch.relu(v)
        v = self.v(v)
        v = torch.tanh(v)

        return pi, v
