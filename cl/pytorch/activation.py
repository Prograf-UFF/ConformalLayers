from .module import ConformalModule
from .utils import EyeTensor
from abc import abstractmethod
from typing import Tuple, Union
import torch


class BaseActivation(ConformalModule):
    def __init__(self):
        super(BaseActivation, self).__init__()

    @property
    @abstractmethod
    def tensors(self) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[EyeTensor, EyeTensor]]:
        pass


class ConformalActivation(BaseActivation):
    def __init__(self):
        super(ConformalActivation, self).__init__()

    def __repr__(self) -> str:
       return "ConformalActivation()"

    @property
    def tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        #TODO To be implemented
        raise NotImplementedError("To be implemented")


class NoActivation(BaseActivation):
    _instance = None

    def __init__(self):
        super(NoActivation, self).__init__()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
       return "NoActivation()"

    @property
    def tensors(self) -> Tuple[EyeTensor, EyeTensor]:
        return EyeTensor(), EyeTensor()
