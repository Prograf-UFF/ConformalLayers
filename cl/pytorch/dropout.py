from .module import ConformalModule
from .utils import EyeTensor
from abc import abstractmethod
from typing import Union
import torch


class BaseDropout(ConformalModule):
    def __init__(self):
        super(BaseDropout, self).__init__()

    @property
    @abstractmethod
    def tensor(self) -> Union[torch.Tensor, EyeTensor]:
        pass


class Dropout(BaseDropout):
    def __init__(self, rate: float):
        super(Dropout, self).__init__()
        self._rate = rate

    def __repr__(self) -> str:
       return "Dropout(rate={})".format(self._rate)

    @property
    def tensor(self) -> torch.Tensor:
        #TODO To be implemented
        raise NotImplementedError("To be implemented")

    @property
    def rate(self) -> float:
        return self._rate


class NoDropout(BaseDropout):
    _instance = None

    def __init__(self):
        super(NoDropout, self).__init__()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
       return "NoDropout()"

    @property
    def tensor(self) -> EyeTensor:
        return EyeTensor()
