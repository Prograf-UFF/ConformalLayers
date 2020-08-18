from .module import ConformalModule
from abc import abstractmethod
from typing import Tuple, Union
import torch


class BasePooling(ConformalModule):
    def __init__(self):
        super(BasePooling, self).__init__()

    @property
    @abstractmethod
    def tensor(self) -> torch.Tensor:
        pass


class NoPooling(BasePooling):
    def __init__(self, kernel_size: Union[int, Tuple[int, ...]]):
        super(NoPooling, self).__init__()

    @property
    def tensor(self) -> torch.Tensor:
        #TODO return torch.eye(n, m)
        raise NotImplementedError("To be implemented")


class MeanPooling(BasePooling):
    def __init__(self, kernel_size: Union[int, Tuple[int, ...]]):
        super(MeanPooling, self).__init__()
        self.__kernel_size = kernel_size

    @property
    def kernel_size(self) -> Union[int, Tuple[int, ...]]:
        return self.__kernel_size

    @property
    def tensor(self) -> torch.Tensor:
        #TODO To be implemented
        raise NotImplementedError("To be implemented")
