import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: list[int, ...], action_space: int):
        super().__init__()
        self.in_features = in_features
        self.feature_dims = hidden_features
        self.action_space = action_space
        layers = []
        feature_dims = [in_features] + hidden_features
        for i in range(len(feature_dims) - 1):
            layers.append(nn.Linear(feature_dims[i], feature_dims[i + 1]))
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

        self.pi = nn.Linear(self.feature_dims[-1], action_space)
        self.v = nn.Linear(self.feature_dims[-1], 1)

    def forward(self, x):
        N, F = x.shape

        r = self.layers(x)
        pi = self.pi(r)
        v = self.v(r)
        v = torch.tanh(v)

        return pi, v


class ResNet(nn.Module):
    """general resnet implementation that will work for most board games"""
    def __init__(self, 
            in_height: int, in_width: int, in_channels: int, residual_blocks: int, action_space: int):
        super().__init__()
        self.in_height = in_height
        self.in_width = in_width
        self.in_channels = in_channels
        self.residual_blocks = residual_blocks
        self.action_space = action_space

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU()
        )
        self.residual_tower = nn.Sequential(*[Block() for _ in range(self.residual_blocks)])

        # policy head
        self.pi_conv_bn = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=2, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=2)
        )
        self.pi = nn.Linear(2 * self.in_height * self.in_width, self.action_space)

        # value head
        self.v_conv_bn = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=1)
        )
        self.v_fc = nn.Linear(self.in_height * self.in_width, 256)
        self.v = nn.Linear(256, 1)

    def forward(self, x):
        N, C, H, W = x.shape

        # residual tower forward
        r = self.conv_block(x)
        r = self.residual_tower(r)

        # policy head forward
        pi = self.pi_conv_bn(r)
        pi = F.relu(pi)
        pi = pi.view(N, 2 * self.in_height * self.in_width)
        pi = self.pi(pi)

        # value head forward
        v = self.v_conv_bn(r)
        v = F.relu(v)
        v = v.view(N, self.in_height * self.in_width)
        v = self.v_fc(v)
        v = F.relu(v)
        v = self.v(v)
        v = torch.tanh(v)

        return pi, v


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=256)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=256)

    def forward(self, x):
        residual = x
        r = self.conv1(x)
        r = self.bn1(r)
        r = F.relu(r)
        r = self.conv2(r)
        r = self.bn2(r)
        r += residual
        r = F.relu(r)
        return r
