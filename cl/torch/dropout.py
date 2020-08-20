from .module import ConformalModule
from .utils import EyeTensor
from abc import abstractmethod
from typing import Optional, Union
import torch


class BaseDropout(ConformalModule):
    def __init__(self, name: Optional[str]=None) -> None:
        super(BaseDropout, self).__init__(name)

    @property
    @abstractmethod
    def tensor(self) -> Union[torch.Tensor, EyeTensor]:
        pass


class Dropout(BaseDropout):
    def __init__(self,
                 rate: float,
                 name: Optional[str]=None) -> None:
        super(Dropout, self).__init__(name)
        self._rate = rate

    def __repr__(self) -> str:
       return "Dropout(rate={}{})".format(self._rate, self._extra_repr(True))

    @property
    def tensor(self) -> torch.Tensor:
        #TODO To be implemented
        raise NotImplementedError("To be implemented")

    @property
    def rate(self) -> float:
        return self._rate


class NoDropout(BaseDropout):
    _instance = None

    def __init__(self) -> None:
        super(NoDropout, self).__init__("NoDropout")

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
       return "NoDropout()"

    @property
    def tensor(self) -> EyeTensor:
        return EyeTensor()
