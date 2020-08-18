from .module import ConformalModule
from abc import abstractmethod
from typing import Tuple, Union
import torch


class BaseConvolution(ConformalModule):
    def __init__(self):
        super(BaseConvolution, self).__init__()

    @property
    @abstractmethod
    def tensor(self) -> torch.Tensor:
        pass


class NoConvolution(BaseConvolution):
    def __init__(self):
        super(NoConvolution, self).__init__()

    @property
    def tensor(self) -> torch.Tensor:
        #TODO return torch.eye(n, m)
        raise NotImplementedError("To be implemented")


class Convolution(BaseConvolution):
    def __init__(self, kernel_size: Union[int, Tuple[int, ...]], bias: bool):
        super(Convolution, self).__init__()
        self.__kernel_size = kernel_size
        self.__bias = bias

    @property
    def bias(self) -> bool:
        return self.__bias

    @property
    def kernel_size(self) -> Union[int, Tuple[int, ...]]:
        return self.__kernel_size

    @property
    def tensor(self) -> torch.Tensor:
        #TODO To be implemented
        raise NotImplementedError("To be implemented")
