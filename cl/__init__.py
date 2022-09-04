__all__ = [
    '__author__', '__author_email__', '__description__', '__url__', '__version__',
    'AvgPool1d', 'AvgPool2d', 'AvgPool3d',
    'ConformalLayers',
    'Conv1d', 'Conv2d', 'Conv3d',
    'Dropout',
    'Flatten',
    'ReSPro',
]

from .about import __author__, __author_email__, __description__, __url__, __version__
from .activation import ReSPro
from .convolution import Conv1d, Conv2d, Conv3d
from .layers import ConformalLayers
from .pooling import AvgPool1d, AvgPool2d, AvgPool3d
from .regularization import Dropout
from .utility import Flatten
