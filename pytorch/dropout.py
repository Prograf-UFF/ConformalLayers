import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

class AbstractDropout(nn.Module):
    def __init__(self):
        super(AbstractDropout, self).__init__()

    def forward(self, x):
        pass

class Dropout(AbstractDropout):
    def __init__(self):
        super(Dropout, self).__init__()

    def forward(self, x):
        pass

class IdentityDropout(AbstractDropout):
    def __init__(self):
        super(IdentityDropout, self).__init__()

    def forward(self, x):
        pass
