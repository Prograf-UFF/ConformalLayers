from .module import ConformalModule
from .utils import EyeTensor, _int_or_size_1_t, _int_or_size_2_t, _int_or_size_3_t, _size_any_t, _pair, _single, _triple
from abc import abstractmethod
from typing import Optional, Union
import torch


class BasePool(ConformalModule):
    def __init__(self, name: Optional[str]=None):
        super(BasePool, self).__init__(name)

    @property
    @abstractmethod
    def tensor(self) -> Union[torch.Tensor, EyeTensor]:
        pass


class AvgPoolNd(BasePool):
    def __init__(self,
                 kernel_size: _size_any_t,
                 name: Optional[str]=None) -> None:
        super(AvgPoolNd, self).__init__(name)
        self._kernel_size = kernel_size

    def __repr__(self) -> str:
       return "AvgPool(kernel_size={}{})".format(self._kernel_size, self._extra_repr(True))

    @property
    def kernel_size(self) -> _size_any_t:
        return self._kernel_size

    @property
    def tensor(self) -> torch.Tensor:
        #TODO To be implemented
        raise NotImplementedError("To be implemented")


class AvgPool1d(AvgPoolNd):
    def __init__(self,
                 kernel_size: _int_or_size_1_t,
                 name: Optional[str]=None) -> None:
        super(AvgPool1d, self).__init__(
            _single(kernel_size),
            name)


class AvgPool2d(AvgPoolNd):
    def __init__(self,
                 kernel_size: _int_or_size_2_t,
                 name: Optional[str]=None) -> None:
        super(AvgPool2d, self).__init__(
            _pair(kernel_size),
            name)


class AvgPool3d(AvgPoolNd):
    def __init__(self,
                 kernel_size: _int_or_size_3_t,
                 name: Optional[str]=None) -> None:
        super(AvgPool3d, self).__init__(
            _triple(kernel_size),
            name)


class NoPool(BasePool):
    _instance = None

    def __init__(self) -> None:
        super(NoPool, self).__init__("NoPool")

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
       return "NoPool()"

    @property
    def tensor(self) -> EyeTensor:
        return EyeTensor()
