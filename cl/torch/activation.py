from .module import ConformalModule
from abc import abstractmethod
from typing import Optional, Tuple
import torch


class SRePro(ConformalModule):
    def __init__(self,
                 name: Optional[str]=None) -> None:
        super(SRePro, self).__init__(name)

    def __repr__(self) -> str:
       return f'SRePro({self._extra_repr(False)})'

    def _output_size(self, in_channels: int, in_volume: Tuple[int, ...]) -> Tuple[int, Tuple[int, ...]]:
        return in_channels, in_volume

    def _register_parent(self, parent, index: int) -> None:
        pass
