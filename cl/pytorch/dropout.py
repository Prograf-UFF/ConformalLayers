from .module import ConformalModule
from abc import abstractmethod
import torch


class BaseDropout(ConformalModule):
    def __init__(self):
        super(BaseDropout, self).__init__()

    @property
    @abstractmethod
    def tensor(self) -> torch.Tensor:
        pass


class NoDropout(BaseDropout):
    def __init__(self):
        super(NoDropout, self).__init__()

    @property
    def tensor(self) -> torch.Tensor:
        #TODO return torch.eye(n, m)
        raise NotImplementedError("To be implemented")


class Dropout(BaseDropout):
    def __init__(self, rate: float):
        super(Dropout, self).__init__()
        self.__rate = rate

    @property
    def tensor(self) -> torch.Tensor:
        #TODO To be implemented
        raise NotImplementedError("To be implemented")

    @property
    def rate(self) -> float:
        return self.__rate
