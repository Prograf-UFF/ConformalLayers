from .decorator import singleton
from .extension import IdentityMatrix, SparseTensor, ZeroTensor
from .module import ConformalModule
from .utils import _size_any_t
from abc import abstractmethod
from collections import OrderedDict
from typing import Optional, Tuple, Union
import math, numpy, torch


class BaseActivation(ConformalModule):
    def __init__(self,
                 *, name: Optional[str]=None) -> None:
        super(BaseActivation, self).__init__(name=name)

    @abstractmethod
    def to_tensor(self, previous: SparseTensor) -> Tuple[Union[SparseTensor, IdentityMatrix], Union[SparseTensor, ZeroTensor]]:
        pass


@singleton
class NoActivation(BaseActivation):
    def output_size(self, in_channels: int, in_volume: _size_any_t) -> Tuple[int, _size_any_t]:
        return in_channels, in_volume

    def to_tensor(self, previous: SparseTensor) -> Tuple[IdentityMatrix, ZeroTensor]:
        nrows, _ = previous.shape
        return IdentityMatrix(nrows, dtype=previous.dtype), ZeroTensor((nrows, nrows, nrows), dtype=previous.dtype)


class SRePro(BaseActivation):
    def __init__(self,
                 alpha: Optional[float]=None,
                 *, name: Optional[str]=None) -> None:
        super(SRePro, self).__init__(name=name)
        self._alpha = float(alpha) if not alpha is None else None

    def _repr_dict(self) -> OrderedDict:
        entries = super()._repr_dict()
        entries['alpha'] = 'Automatic' if self._alpha is None else self._alpha
        return entries

    def output_size(self, in_channels: int, in_volume: _size_any_t) -> Tuple[int, _size_any_t]:
        return in_channels, in_volume

    def to_tensor(self, previous: SparseTensor) -> Tuple[SparseTensor, SparseTensor]:
        nrows, _ = previous.shape
        ind = numpy.arange(nrows, dtype=numpy.int64)
        # Compute the alpha parameter
        if self._alpha is None:
            symmetric = torch.mm(previous, previous.t())
            alpha = torch.sqrt(math.sqrt(symmetric.nnz) * symmetric.values[:-1, ...].max(0, keepdim=True)[0]) # We use symmetric.values[:-1, ...] to skeep the homogeneous coordinate (it is always 1 and does not affect the transformed vector).
        else:
            alpha = torch.as_tensor((self.alpha,), dtype=previous.dtype)
        # Compute the non-constant coefficient of the matrix
        matrix_values = torch.ones((nrows,), dtype=previous.dtype)
        matrix_values[-1] = 0.5 * alpha
        matrix = SparseTensor((ind, ind), matrix_values, (nrows, nrows), coalesced=True)
        # Make rank-3 tensor
        max_ind = numpy.full((nrows - 1,), nrows - 1, dtype=numpy.int64)
        tensor_values = (1 / (2 * alpha)).expand(nrows - 1)
        tensor = SparseTensor((max_ind, ind[:-1], ind[:-1]), tensor_values, (nrows, nrows, nrows), coalesced=True)
        # Return tensor representation of the activation function
        return matrix, tensor
    
    @property
    def alpha(self) -> Optional[float]:
        return self._alpha
