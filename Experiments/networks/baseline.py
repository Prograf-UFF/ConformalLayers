import cl
import torch
import torch.nn as nn


class BaseReSProNet(nn.Module):

    def __init__(self) -> None:
        super(BaseReSProNet, self).__init__()
        self.features = cl.ConformalLayers(
            cl.Conv2d(3, 32, kernel_size=3),
            cl.ReSPro(),
            cl.AvgPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(7200, 10)

    def forward(self, x) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class BaseLinearNet(nn.Module):

    def __init__(self) -> None:
        super(BaseLinearNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            # Linear activation
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(7200, 10)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class BaseReLUNet(nn.Module):
    
    def __init__(self) -> None:
        super(BaseReLUNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(7200, 10)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
