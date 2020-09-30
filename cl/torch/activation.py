from .module import ConformalModule
from .utils import _size_any_t
from typing import Optional, Tuple


class SRePro(ConformalModule):
    def __init__(self,
                 name: Optional[str]=None) -> None:
        super(SRePro, self).__init__(name)

    def __repr__(self) -> str:
       return f'SRePro({self._extra_repr(False)})'

    def _output_size(self, in_channels: int, in_volume: _size_any_t) -> Tuple[int, _size_any_t]:
        return in_channels, in_volume
