from abc import ABC, abstractmethod
from typing import Optional


class ConformalModule(ABC):
    def __init__(self, name: Optional[str]=None) -> None:
        super(ConformalModule, self).__init__()
        self._name = name

    def _extra_repr(self, comma: bool) -> str:
        return "" if self._name is None else "{}name={}".format(", " if comma else "", self._name)

    def _register_parent(self, parent, layer: int) -> None:
        pass

    @property
    def name(self) -> Optional[str]:
        return self._name
