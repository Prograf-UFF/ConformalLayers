from .decorator import singleton
from .extension import IdentityMatrix, ZeroTensor
from .module import ConformalModule
from .utils import _size_any_t
from abc import abstractmethod
from typing import Optional, Tuple, Union
import numpy, torch


class BaseActivation(ConformalModule):
    def __init__(self,
                 name: Optional[str]=None) -> None:
        super(BaseActivation, self).__init__(name)

    def __repr__(self) -> str:
       return f'{self.__class__.__name__}({self._extra_repr(False)})'

    @abstractmethod
    def to_tensor(self, previous: torch.Tensor) -> Tuple[Union[torch.Tensor, IdentityMatrix], Union[torch.Tensor, ZeroTensor]]:
        pass


@singleton
class NoActivation(BaseActivation):
    def to_tensor(self, previous: torch.Tensor) -> Tuple[IdentityMatrix, ZeroTensor]:
        nrows, _ = previous.shape
        return IdentityMatrix(nrows, dtype=previous.dtype), ZeroTensor((nrows, nrows, nrows), dtype=previous.dtype)

    def output_size(self, in_channels: int, in_volume: _size_any_t) -> Tuple[int, _size_any_t]:
        return in_channels, in_volume


class SRePro(BaseActivation):
    def __init__(self,
                 name: Optional[str]=None) -> None:
        super(SRePro, self).__init__(name)

    def to_tensor(self, previous: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        nrows, _ = previous.shape
        range_coords = torch.arange(nrows, dtype=torch.int32)
        # Compute the alpha parameter
        alpha = previous.max() #TODO Parei aqui!
        # Compute the non-constant coefficient of the matrix
        matrix_feats = torch.ones((nrows,), dtype=previous.dtype)
        matrix_feats[-1] = 0.5 * alpha
        matrix = torch.sparse_coo_tensor((range_coords, range_coords), matrix_feats, size=(nrows, nrows), dtype=previous.dtype)
        # Make rank-3 tensor
        max_coords = torch.full((nrows - 1,), nrows - 1, dtype=torch.int32)
        tensor_feats = torch.full((nrows - 1,), 1 / (2 * alpha), dtype=previous.dtype)
        tensor = torch.sparse_coo_tensor((max_coords, range_coords[:-1], range_coords[:-1]), tensor_feats, size=(nrows, nrows, nrows), dtype=previous.dtype)
        # Return tensor representation of the activation function
        return matrix, tensor
    
    def output_size(self, in_channels: int, in_volume: _size_any_t) -> Tuple[int, _size_any_t]:
        return in_channels, in_volume
