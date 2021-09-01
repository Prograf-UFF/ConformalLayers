import torch
import torch.nn as nn
import torch.nn.functional as func
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import cl


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.features = cl.ConformalLayers(
            cl.Conv2d(3, 6, 5),
            cl.ReSPro(),
            cl.AvgPool2d(kernel_size=2, stride=2),
            cl.Conv2d(6, 16, 5),
            cl.ReSPro(),
            cl.AvgPool2d(kernel_size=2, stride=2),
        )

        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LeNetCL(nn.Module):
    def __init__(self):
        super(LeNetCL, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = func.relu(self.conv1(x))
        x = func.max_pool2d(x, 2)
        x = func.relu(self.conv2(x))
        x = func.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        return x
