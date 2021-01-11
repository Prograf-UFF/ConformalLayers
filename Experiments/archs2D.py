import torch.nn as nn
import torch.nn.functional as F
import os
import sys

try:
    import cl
except ModuleNotFoundError:
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    import cl


class CLSaturday(nn.Module):
    """
    Saturday Architecture implemented in Conformal Layers

    """

    def __init__(self):
        super(CLSaturday, self).__init__()
        self.features = cl.ConformalLayers(
            cl.Conv2d(in_channels=1, out_channels=128, kernel_size=5),
            cl.Dropout(),
            cl.AvgPool2d(kernel_size=2, stride=2),
            cl.SRePro(),
          )
        self.fc1 = nn.Linear(18432, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out

class CLLenet(nn.Module):
    """
    LeNet Architecture implemented in Conformal Layers
    
    """

    def __init__(self):
        super(CLLenet, self).__init__()
        self.features = cl.ConformalLayers(
            cl.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            cl.SRePro(),
            cl.AvgPool2d(kernel_size=2, stride=2),

            cl.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            cl.SRePro(),
            cl.AvgPool2d(kernel_size=2, stride=2),
          )
        self.fc1 = nn.Linear(256, 120)
        self.activ1 = cl.ConformalLayers(cl.SRePro())
        self.fc2 = nn.Linear(120, 84)
        self.activ2 = cl.ConformalLayers(cl.SRePro())
        self.fc3 = nn.Linear(84, 10)
        self.activ3 = cl.ConformalLayers(cl.SRePro())

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.activ1(out.expand(1, *out.shape).permute(*range(1, len(out.shape) + 1), 0))
        out = self.fc2(out[:, :, 0])
        out = self.activ2(out.expand(1, *out.shape).permute(*range(1, len(out.shape) + 1), 0))
        out = self.fc3(out[:, :, 0])
        out = self.activ3(out.expand(1, *out.shape).permute(*range(1, len(out.shape) + 1), 0))
        return out[:, :, 0]

class DefaultLeNet(nn.Module):
    """
    LeNet Architecture implemented in native torch

    """
    
    def __init__(self):
        super(DefaultLeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, bias=False),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, bias=False),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
          )
        self.fc1 = nn.Linear(256, 120)
        self.activ1 = nn.Tanh()
        self.fc2 = nn.Linear(120, 84)
        self.activ2 = nn.Tanh()
        self.fc3 = nn.Linear(84, 10)
        self.activ3 = nn.Softmax()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.shape[0], -1)
        out = self.fc1(out)
        out = self.activ1(out)
        out = self.fc2(out)
        out = self.activ2(out)
        out = self.fc3(out)
        out = self.activ3(out)
        return out