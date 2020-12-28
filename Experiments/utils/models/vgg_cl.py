import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os

sys.path.insert(1, '../../..')

try:
    import cl
except ModuleNotFoundError:
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    import cl


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = cl.ConformalLayers(
            cl.Conv2d(in_channels=1, out_channels=64, kernel_size=5),
            cl.AvgPool2d(kernel_size=2, stride=2),
            cl.SRePro(),
            # cl.Conv2d(in_channels=64, out_channels=64, kernel_size=5),
            # cl.SRePro(),
            # cl.AvgPool2d(kernel_size=2, stride=2)
          )
        # self.fc1 = nn.Linear(800, 10, bias=False)
        self.fc1 = nn.Linear(9216, 10, bias=False)
        # if os.path.exists('kernelConv.pt'):
        #     print("Loading existing kernel...")
        #     self.features[0]._kernel = torch.load('kernelConv.pt')
        #     self.fc1.weight = torch.load('kernelLinear.pt')
        #     print("Done!")
        # else:
        #     print("Saving weights...")
        #     torch.save(self.features[0].kernel, 'kernelConv.pt')
        #     torch.save(self.fc1.weight, 'kernelLinear.pt')
        #     print("Done!")
        # self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        out = self.features(x)
        out = nn.Flatten()(out) #flatten
        # print(out.shape)
        out = self.fc1(out)
        # out = out.clamp(min=1e-4)
        # out = self.fc2(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 1
        for x in cfg:
            if x == 'M':
                layers.append(cl.AvgPool2d(kernel_size=2, stride=2, padding=0))
            else:
                layers.append(cl.Conv2d(in_channels=in_channels, out_channels=x, kernel_size=3, padding=1))
                layers.append(cl.SRePro())
                break
                
                in_channels = x
        layers.append(cl.AvgPool2d(kernel_size=1, stride=1))
        return cl.ConformalLayers(*layers)
