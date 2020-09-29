from abc import ABC, abstractmethod
from typing import Optional, Tuple


class ConformalModule(ABC):
    def __init__(self, name: Optional[str]=None) -> None:
        super(ConformalModule, self).__init__()
        self._name = name

    def _extra_repr(self, comma: bool) -> str:
        return '' if self._name is None else f'{", " if comma else ""}name={self._name}'

    @abstractmethod
    def _output_size(self, in_channels: int, in_volume: Tuple[int, ...]) -> Tuple[int, Tuple[int, ...]]:
        pass
    
    @abstractmethod
    def _register_parent(self, parent, index: int) -> None:
        pass

    @property
    def name(self) -> Optional[str]:
        return self._name
