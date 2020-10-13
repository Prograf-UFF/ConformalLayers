from .decorator import singleton
from .extension import IdentityMatrix, SparseTensor, ZeroTensor
from .module import ConformalModule
from .utils import _size_any_t
from abc import abstractmethod
from typing import Optional, Tuple, Union
import numpy, torch


class BaseActivation(ConformalModule):
    def __init__(self,
                 name: Optional[str]=None) -> None:
        super(BaseActivation, self).__init__(name)

    @abstractmethod
    def to_tensor(self, previous: SparseTensor) -> Tuple[Union[SparseTensor, IdentityMatrix], Union[SparseTensor, ZeroTensor]]:
        pass


@singleton
class NoActivation(BaseActivation):
    def __repr__(self) -> str:
       return f'{self.__class__.__name__}({self._extra_repr(False)})'

    def to_tensor(self, previous: SparseTensor) -> Tuple[IdentityMatrix, ZeroTensor]:
        nrows, _ = previous.shape
        return IdentityMatrix(nrows, dtype=previous.dtype), ZeroTensor((nrows, nrows, nrows), dtype=previous.dtype)

    def output_size(self, in_channels: int, in_volume: _size_any_t) -> Tuple[int, _size_any_t]:
        return in_channels, in_volume


class SRePro(BaseActivation):
    def __init__(self,
                 alpha: Optional[float]=None,
                 name: Optional[str]=None) -> None:
        super(SRePro, self).__init__(name)
        self._alpha = alpha

    def __repr__(self) -> str:
       return f'{self.__class__.__name__}(alpha={self.alpha}{self._extra_repr(True)})'

    def to_tensor(self, previous: SparseTensor) -> Tuple[SparseTensor, SparseTensor]:
        nrows, _ = previous.shape
        ind = numpy.arange(nrows, dtype=numpy.int64)
        # Compute the alpha parameter
        if self._alpha is None:
            alpha = previous.max() #TODO Parei aqui!
        else:
            alpha = self.alpha
        # Compute the non-constant coefficient of the matrix
        matrix_values = torch.ones((nrows,), dtype=previous.dtype)
        matrix_values[-1] = 0.5 * alpha
        matrix = SparseTensor((ind, ind), matrix_values, (nrows, nrows), coalesced=True)
        # Make rank-3 tensor
        max_ind = numpy.full((nrows - 1,), nrows - 1, dtype=numpy.int64)
        tensor_values = torch.full((nrows - 1,), 1 / (2 * alpha), dtype=previous.dtype)
        tensor = SparseTensor((max_ind, ind[:-1], ind[:-1]), tensor_values, (nrows, nrows, nrows), coalesced=True)
        # Return tensor representation of the activation function
        return matrix, tensor
    
    def output_size(self, in_channels: int, in_volume: _size_any_t) -> Tuple[int, _size_any_t]:
        return in_channels, in_volume

    @property
    def alpha(self):
        return self._alpha
