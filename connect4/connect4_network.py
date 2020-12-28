import torch
import torch.nn as nn

class Connect4_Model(nn.Module):
    def __init__(self, args):
        super(Connect4_Model, self).__init__()
        self.args = args
        self.board_width, self.board_height = 7, 6

        self.conv_block1 = self.conv_block(1, self.args.num_channels, kernel_size=3, stride=1, padding=self.same_padding(3))
        self.conv_block2 = self.conv_block(self.args.num_channels, self.args.num_channels, kernel_size=3, stride=1, padding=self.same_padding(3))
        self.conv_block3 = self.conv_block(self.args.num_channels, self.args.num_channels, kernel_size=3, stride=1)
        self.conv_block4 = self.conv_block(self.args.num_channels, self.args.num_channels, kernel_size=3, stride=1)

        self.width_size_out = self.conv_size_out(self.conv_size_out(7, 3, 1), 3, 1)
        self.height_size_out = self.conv_size_out(self.conv_size_out(6, 3, 1), 3, 1)

        self.fc1 = self.linear_block(self.args.num_channels * self.width_size_out * self.height_size_out, 1024)
        self.fc2 = self.linear_block(1024, 512)

        self.pi = nn.Linear(512, 7)
        self.v = nn.Linear(512, 1)
        
    def forward(self, b):
        b = b.reshape(-1, 1, self.board_width, self.board_height)
        r = self.conv_block1(b)
        r = self.conv_block2(r)
        r = self.conv_block3(r)
        r = self.conv_block4(r)
        r = r.reshape(b.shape[0], -1)
        r = self.fc1(r)
        r = self.fc2(r)

        pi = torch.log_softmax(self.pi(r), dim=1)
        v = torch.tanh(self.v(r))
        return pi, v

    def same_padding(self, kernel_size):
        return kernel_size // 2

    def conv_size_out(self, size, kernel_size, stride, padding=0):
        size += padding*2
        return (size - (kernel_size - 1) - 1) // stride + 1

    def conv_block(self, in_channels, out_channels, *args, **kwargs):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, *args, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def linear_block(self, in_features, out_features):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU()
        )