import torch
import torch.nn as nn
import torch.nn.functional as F

class Connect4_Model(nn.Module):
    def __init__(self, args):
        super(Connect4_Model, self).__init__()
        self.args = args
        self.board_height, self.board_width = 6, 7

        self.conv_block1 = self.conv_block(1, 512, kernel_size=3, stride=1, padding=self.same_padding(3))
        self.conv_block2 = self.conv_block(512, 512, kernel_size=3, stride=1, padding=self.same_padding(3))
        self.conv_block3 = self.conv_block(512, 512, kernel_size=3, stride=1, padding=self.same_padding(3))
        self.conv_block4 = self.conv_block(512, 512, kernel_size=3, stride=1)
        self.conv_block5 = self.conv_block(512, 512, kernel_size=3, stride=1)

        self.height_size_out = self.conv_size_out(self.conv_size_out(self.board_height, 3, 1), 3, 1)
        self.width_size_out = self.conv_size_out(self.conv_size_out(self.board_width, 3, 1), 3, 1)

        self.fc1 = self.linear_block(512 * self.height_size_out * self.width_size_out, 1024)
        self.fc2 = self.linear_block(1024, 512)

        self.pi = nn.Linear(512, self.board_width)
        self.value = nn.Linear(512, 1)
        
    def forward(self, b):
        b = b.reshape(-1, 1, self.board_height, self.board_width)
        r = self.conv_block1(b)
        r = self.conv_block2(r)
        r = self.conv_block3(r)
        r = self.conv_block4(r)
        r = self.conv_block5(r)
        r = r.reshape(b.shape[0], -1)
        r = self.fc1(r)
        r = self.fc2(r)

        pi = self.pi(r)
        value = torch.tanh(self.value(r))
        return pi, value

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


class Policy_Head(nn.Module):
    def __init__(self, in_channels, in_height, in_width):
        super(Policy_Head, self).__init__()

        self.conv = nn.Conv2d(in_channels, 2, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(2)
        self.pi = nn.Linear(2 * in_height * in_width, 7)

    def forward(self, r):
        pi = torch.relu(self.bn(self.conv(r)))
        pi = pi.reshape(pi.shape[0], -1)
        pi = self.pi(pi)
        
        return pi


class Value_Head(nn.Module):
    def __init__(self, in_channels, in_height, in_width):
        super(Value_Head, self).__init__()

        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(1)
        self.fc = nn.Linear(in_height * in_width, 256)
        self.v = nn.Linear(256, 1)

    def forward(self, r):
        v = torch.relu(self.bn(self.conv(r)))
        v = v.reshape(v.shape[0], -1)
        v = torch.relu(self.fc(v))
        v = torch.tanh(self.v(v))

        return v


class Dual_Conv(nn.Module):
    """
    'dual-conv' network mimicked from AlphaGo Zero paper.
    """
    def __init__(self, args):
        super(Dual_Conv, self).__init__()
        self.args = args

        self.conv = self.conv_block(1, 256, kernel_size=3, stride=1, padding=1)
        self.conv_blocks = [self.conv_block(256, 256, kernel_size=3, stride=1, padding=1) for _ in range(4)]

        self.pi = self.policy_head()
        self.v = self.value_head()

    def forward(self, b):
        r = self.conv(b)
        for conv_block in self.conv_blocks:
            r = conv_block(r)

        pi = self.pi(r)
        v = self.v(r)

        return pi, v

    def conv_block(self, in_channels, out_channels, *args, **kwargs):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, *args, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def policy_head(self):
        return Policy_Head(256, 6, 7)

    def value_head(self):
        return Value_Head(256, 6, 7)

    def to_device(self, device):
        for block in self.conv_blocks:
            block.to(device)
            self.pi.to(device)
            self.v.to(device)
            self.to(device)


class Connect4_ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(Connect4_ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = self.linear_block(512*block.expansion, 2048)
        
        self.pi = nn.Linear(2048, 7)
        self.v = nn.Linear(2048, 1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def linear_block(self, in_features, out_features):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU()
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        pi = self.pi(out)
        v = self.v(out)

        return pi, torch.tanh(v)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out