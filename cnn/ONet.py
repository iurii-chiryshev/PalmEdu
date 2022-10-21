import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.transpose(3, 2).contiguous()
        return x.view(x.size(0), -1)


class ONet(nn.Module):
    def __init__(self):
        super(ONet, self).__init__()

        self.features = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(3, 32, 3, 1)),
                ('prelu1', nn.PReLU(32)),
                ('pool1', nn.MaxPool2d(3, 2, ceil_mode=True)),
                ('conv2', nn.Conv2d(32, 64, 3, 1)),
                ('prelu2', nn.PReLU(64)),
                ('pool2', nn.MaxPool2d(3, 2, ceil_mode=True)),
                ('conv3', nn.Conv2d(64, 64, 3, 1)),
                ('prelu3', nn.PReLU(64)),
                ('pool3', nn.MaxPool2d(2, 2, ceil_mode=True)),
                ('conv4', nn.Conv2d(64, 128, 2, 1)),
                ('prelu4', nn.PReLU(128)),
            ]))

        self.flatten = Flatten()
        self.conv5 = nn.Linear(3200, 256)  # 21632 if [128x128], 3200 if [64x64]
        self.drop5 = nn.Dropout(0.25)
        self.prelu5 = nn.PReLU(256)
        self.result = nn.Linear(256, 1) #

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.conv5(x)
        x = self.drop5(x)

        x = self.result(x).view(-1)
        return x


if __name__ == '__main__':
    net = ONet()
    dummy_input = torch.randn(1, 3, 64, 64)
    out = net(dummy_input)
    print(out)
