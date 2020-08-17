import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

class AbstractPooling(nn.Module):
    def __init__(self):
        super(AbstractPooling, self).__init__()

    def forward(self, x):
        pass

class Pooling(AbstractPooling):
    def __init__(self):
        super(Pooling, self).__init__()

    def forward(self, x):
        pass

class IdentityPooling(AbstractPooling):
    def __init__(self):
        super(IdentityPooling, self).__init__()

    def forward(self, x):
        pass
