import torch
import torch.nn as nn
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import cl


class DkNetCL(nn.Module):
    def __init__(self, depth):
        super(DkNetCL, self).__init__()
        f = []
        input_channels = 3
        filters = 32
        for i in range(depth):
            f.append(cl.Conv2d(input_channels, filters, 3, padding=1))
            f.append(cl.SRePro())
            input_channels = filters

        self.features = cl.ConformalLayers(*f)

        flattened_size = 32768
        self.fc1 = nn.Linear(flattened_size, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        return x


class DkNet(nn.Module):
    def __init__(self, depth):
        super(DkNet, self).__init__()
        f = []
        input_channels = 3
        filters = 32
        for i in range(depth):
            f.append(nn.Conv2d(input_channels, filters, 3, padding=1))
            f.append(nn.ReLU())
            input_channels = filters

        self.features = nn.Sequential(*f)

        flattened_size = 32768
        self.fc1 = nn.Linear(flattened_size, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        return x


class D3ModNetCL(nn.Module):
    def __init__(self):
        super(D3ModNetCL, self).__init__()
        f = []
        input_channels = 3
        filters = 32
        for i in range(3):
            f.append(cl.Conv2d(input_channels, filters, 3))
            f.append(cl.SRePro())
            f.append(cl.AvgPool2d(2, 2))
            input_channels = filters

        self.features = cl.ConformalLayers(*f)
        self.fc1 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        return x


class D3ModNet(nn.Module):
    def __init__(self):
        super(D3ModNet, self).__init__()
        f = []
        fp = []
        input_channels = 3
        filters = 32
        for i in range(3):
            f.append(nn.Conv2d(input_channels, filters, 3))
            f.append(nn.ReLU())
            f.append(nn.AvgPool2d(2, 2))
            input_channels = filters

        self.features = nn.Sequential(*f)
        self.fc1 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        return x
