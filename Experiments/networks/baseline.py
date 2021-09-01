import torch
import torch.nn as nn
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import cl


class BaseReSProNet(nn.Module):
    def __init__(self) -> None:
        super(BaseReSProNet, self).__init__()
        self.features = cl.ConformalLayers(
            cl.Conv2d(3, 32, kernel_size=5),
            cl.ReSPro(),
            cl.AvgPool2d(kernel_size=3, stride=3),
        )
        self.fc = nn.Linear(2592, 10)

    def forward(self, x) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class BaseLinearNet(nn.Module):
    def __init__(self) -> None:
        super(BaseLinearNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5),
            # Linear activation
            nn.AvgPool2d(kernel_size=3, stride=3),
        )
        self.fc = nn.Linear(2592, 10)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class BaseReLUNet(nn.Module):
    def __init__(self) -> None:
        super(BaseReLUNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, stride=3),
        )
        self.fc = nn.Linear(2592, 10)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
