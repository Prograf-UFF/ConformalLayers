import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

class AbstractActivation(nn.Module):
    def __init__(self):
        super(AbstractActivation, self).__init__()

    def forward(self, x):
        pass

class Activation(AbstractActivation):
    def __init__(self):
        super(Activation, self).__init__()

    def forward(self, x):
        pass

class IdentityActivation(AbstractActivation):
    def __init__(self):
        super(IdentityActivation, self).__init__()

    def forward(self, x):
        pass
