from .decorator import singleton
from .extension import SparseTensor
from .module import ConformalModule
from .utils import SizeAny
from abc import abstractmethod
from collections import OrderedDict
from typing import Optional, Tuple
import math, numpy, torch


class BaseActivation(ConformalModule):
    def __init__(self,
                 *, name: Optional[str]=None) -> None:
        super(BaseActivation, self).__init__(None, name=name)

    @abstractmethod
    def to_tensor(self, previous: SparseTensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        pass


class NoActivation(BaseActivation):
    def __init__(self) -> None:
        super(NoActivation, self).__init__()

    def to_tensor(self, previous: SparseTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        matrix_scalar = torch.as_tensor(1, dtype=previous.dtype, device=previous.device)
        tensor_scalar = None
        return matrix_scalar, tensor_scalar


class SRePro(BaseActivation):
    def __init__(self,
                 alpha: Optional[float]=None,
                 *, name: Optional[str]=None) -> None:
        super(SRePro, self).__init__(name=name)
        self._alpha = alpha if alpha is not None else None

    def _repr_dict(self) -> OrderedDict:
        entries = super()._repr_dict()
        entries['alpha'] = 'Automatic' if self._alpha is None else self._alpha
        return entries

    def output_size(self, in_channels: int, in_volume: SizeAny) -> Tuple[int, SizeAny]:
        return in_channels, in_volume

    def to_tensor(self, previous: SparseTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute the alpha parameter
        if self._alpha is None:
            # print("before mm")
            symmetric = torch.mm(previous, previous.t())
            # print("after mm")
            alpha = torch.sqrt(math.sqrt(symmetric.nnz) * symmetric.values.detach().abs().max(0)[0])
        else:
            alpha = torch.as_tensor(self.alpha, dtype=previous.dtype, device=previous.device)
        # Compute the last coefficient of the matrix
        matrix_scalar = alpha / 2
        # Compute the coefficient on the main diagonal of the last slice of the tensor
        tensor_scalar = 1 / (2 * alpha)
        # Return the scalars of the tensor representation of the activation function
        return matrix_scalar, tensor_scalar
    
    @property
    def alpha(self) -> Optional[float]:
        return self._alpha
