from .module import ConformalModule
from .utils import EyeTensor
from abc import abstractmethod
from typing import Optional, Tuple, Union
import torch


class BaseActivation(ConformalModule):
    def __init__(self, name: Optional[str]=None) -> None:
        super(BaseActivation, self).__init__(name)

    @property
    @abstractmethod
    def tensors(self) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[EyeTensor, EyeTensor]]:
        pass


class SRePro(BaseActivation):
    def __init__(self,
                 name: Optional[str]=None) -> None:
        super(SRePro, self).__init__(name)

    def __repr__(self) -> str:
       return "SRePro({})".format(self._extra_repr(False))

    @property
    def tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        #TODO To be implemented
        raise NotImplementedError("To be implemented")


class NoActivation(BaseActivation):
    _instance = None

    def __init__(self) -> None:
        super(NoActivation, self).__init__("NoActivation")

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
       return "NoActivation()"

    @property
    def tensors(self) -> Tuple[EyeTensor, EyeTensor]:
        return tuple(EyeTensor(), EyeTensor())
