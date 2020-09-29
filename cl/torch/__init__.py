from .activation import SRePro
from .convolution import Conv1d, Conv2d, Conv3d, ConvNd, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d, ConvTransposeNd
from .layers import ConformalLayers
from .pooling import AvgPool1d, AvgPool2d, AvgPool3d, AvgPoolNd
#TODO Que outros módulos podem ser incluídos?

__all__ = [
    'AvgPool1d', 'AvgPool2d', 'AvgPool3d', 'AvgPoolNd',
    'ConformalLayers',
    'Conv1d', 'Conv2d', 'Conv3d', 'ConvNd', 'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d', 'ConvTransposeNd',
    'SRePro'
]
