from .module import ConformalModule
from .utils import EyeTensor
from abc import abstractmethod
from typing import Tuple, Union
import torch


class BaseConvolution(ConformalModule):
    def __init__(self):
        super(BaseConvolution, self).__init__()

    @property
    @abstractmethod
    def tensor(self) -> Union[torch.Tensor, EyeTensor]:
        pass


class Convolution(BaseConvolution):
    def __init__(self, kernel_size: Union[int, Tuple[int, ...]], bias: bool):
        super(Convolution, self).__init__()
        self._kernel_size = kernel_size
        self._bias = bias

    def __repr__(self) -> str:
       return "Convolution(kernel_size={}, bias={})".format(self._kernel_size, self._bias)

    @property
    def bias(self) -> bool:
        return self._bias

    @property
    def kernel_size(self) -> Union[int, Tuple[int, ...]]:
        return self._kernel_size

    @property
    def tensor(self) -> torch.Tensor:
        #TODO To be implemented
        raise NotImplementedError("To be implemented")


class NoConvolution(BaseConvolution):
    _instance = None

    def __init__(self):
        super(NoConvolution, self).__init__()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
       return "NoConvolution()"

    @property
    def tensor(self) -> EyeTensor:
        return EyeTensor()
