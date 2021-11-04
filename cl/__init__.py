__all__ = [
    'AvgPool1d', 'AvgPool2d', 'AvgPool3d',
    'ConformalLayers',
    'Conv1d', 'Conv2d', 'Conv3d',
    'Dropout',
    'Flatten',
    'SRePro',
]
__author__ = 'Eduardo V. Sousa, Leandro A. F. Fernandes, Cristina N. Vasconcelos'
__author_email__ = 'eduardovera@ic.uff.br, laffernandes@ic.uff.br, crisnv@ic.uff.br'
__version__ = '1.1.0'

from .activation import ReSPro
from .convolution import Conv1d, Conv2d, Conv3d
from .layers import ConformalLayers
from .pooling import AvgPool1d, AvgPool2d, AvgPool3d
from .regularization import Dropout
from .utility import Flatten
