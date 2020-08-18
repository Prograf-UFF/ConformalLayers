from .module import ConformalModule
from abc import abstractmethod
from typing import Tuple
import torch


class BaseActivation(ConformalModule):
    def __init__(self):
        super(BaseActivation, self).__init__()

    @property
    @abstractmethod
    def tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


class NoActivation(BaseActivation):
    def __init__(self):
        super(NoActivation, self).__init__()

    @property
    def tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        #TODO return torch.eye(n, m), something
        raise NotImplementedError("To be implemented")


class ConformalActivation(BaseActivation):
    def __init__(self):
        super(ConformalActivation, self).__init__()

    @property
    def tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        #TODO To be implemented
        raise NotImplementedError("To be implemented")
