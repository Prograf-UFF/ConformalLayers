'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn


def activate(v, alpha):
    batches, in_channels, *in_volume = v.shape
    # verificar a substituição da einsum por uma versão do dot
    dot = torch.einsum('bcd,bcd->bc', v.view(batches, in_channels, -1), v.view(batches, in_channels, -1))
    dot = dot.expand(1, 1, dot.shape[0], dot.shape[1]).permute(2, 3, 0, 1)
    no = (dot / (2*alpha)) + (.5*alpha)
    
    return v / no

class ConformalActivation(nn.Module):

    def __init__(self, alpha):
        super().__init__() # init the base class
        self.alpha = alpha

    def forward(self, v):      
        return activate(v, self.alpha)


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        
        batches, in_channels, *in_volume = x.shape
        
        xo = x.view(batches, -1).norm(dim=(1))        
        xo = xo.expand(1, 1, 1, xo.shape[0]).permute(3, 0, 1, 2)
        x = x / xo
#============================ HERE
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
