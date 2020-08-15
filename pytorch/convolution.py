import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

class AbstractConvolution(nn.Module):
    def __init__(self):
        super(AbstractConvolution, self).__init__()

    def forward(self, x):
        pass

class Convolution(AbstractConvolution):
    def __init__(self):
        super(Convolution, self).__init__()

    def forward(self, x):
        pass

class IdentityConvolution(AbstractConvolution):
    def __init__(self):
        super(IdentityConvolution, self).__init__()

    def forward(self, x):
        pass
