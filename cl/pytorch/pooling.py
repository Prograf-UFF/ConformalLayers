from .module import ConformalModule
from .utils import EyeTensor
from abc import abstractmethod
from typing import Tuple, Union
import torch


class BasePooling(ConformalModule):
    def __init__(self):
        super(BasePooling, self).__init__()

    @property
    @abstractmethod
    def tensor(self) -> Union[torch.Tensor, EyeTensor]:
        pass


class AveragePooling(BasePooling):
    def __init__(self, kernel_size: Union[int, Tuple[int, ...]]):
        super(AveragePooling, self).__init__()
        self._kernel_size = kernel_size

    def __repr__(self) -> str:
       return "AveragePooling(kernel_size={})".format(self._kernel_size)

    @property
    def kernel_size(self) -> Union[int, Tuple[int, ...]]:
        return self._kernel_size

    @property
    def tensor(self) -> torch.Tensor:
        #TODO To be implemented
        raise NotImplementedError("To be implemented")


class NoPooling(BasePooling):
    _instance = None

    def __init__(self):
        super(NoPooling, self).__init__()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
       return "NoPooling()"

    @property
    def tensor(self) -> EyeTensor:
        return EyeTensor()
