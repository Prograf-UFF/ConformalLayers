from .decorator import singleton
from abc import ABC, abstractmethod
from typing import Callable, Iterable, Optional, Tuple, Union
import functools, torch


HANDLED_FUNCTIONS = {}


def implements(torch_function: Callable):
    @functools.wraps(torch_function)
    def decorator(func):
        HANDLED_FUNCTIONS[torch_function] = func
        return func
    return decorator


class CustomTensor(ABC):
    def __init__(self, size: Iterable[int], dtype: torch.dtype) -> None:
        super(CustomTensor, self).__init__()
        self._shape = torch.Size(size)
        self._dtype = dtype

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    @abstractmethod
    def to_sparse(self) -> torch.Tensor:
        pass
    
    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def shape(self) -> torch.Size:
        return self._shape


class IdentityMatrix(CustomTensor):
    def __init__(self, size: int, dtype: torch.dtype) -> None:
        super(IdentityMatrix, self).__init__((size, size), dtype)

    def __repr__(self) -> str:
        return f'IdentityMatrix(size={(*self.shape,)}, dtype={self.dtype})'

    def to_sparse(self) -> torch.Tensor:
        coords = torch.range(self.shape[0], dtype=torch.uint32)
        return torch.sparse_coo_tensor((coords, coords), torch.ones((self.shape[0],), dtype=self.dtype), size=self.shape, dtype=self.dtype)


class ZeroTensor(CustomTensor):
    def __init__(self, size: Iterable[int], dtype: torch.dtype) -> None:
        super(ZeroTensor, self).__init__(size, dtype)

    def __repr__(self) -> str:
        return f'ZeroTensor(size={(*self.shape,)}, dtype={self.dtype})'

    def to_sparse(self) -> torch.Tensor:
        return torch.sparse_coo_tensor(size=self.shape, dtype=self.dtype)


@implements(torch.add)
def _add(lhs: Union[torch.Tensor, CustomTensor], rhs: Union[torch.Tensor, CustomTensor], out: Optional[torch.Tensor]=None) -> Union[torch.Tensor, CustomTensor]:
    if lhs.shape != rhs.shape or (not out is None and lhs.shape != out.shape):
        raise ValueError('Shape mismatch.')
    if isinstance(lhs, IdentityMatrix):
        if isinstance(rhs, ZeroTensor):
            return _try_copy(lhs, out)
        if isinstance(rhs, torch.Tensor):
            return torch.add(lhs.to_sparse(), rhs, out)
    elif isinstance(lhs, ZeroTensor):
        return _try_copy(rhs, out)
    elif isinstance(lhs, torch.Tensor):
        if isinstance(rhs, IdentityMatrix):
            return torch.add(lhs, rhs.to_sparse(), out)
        if isinstance(rhs, ZeroTensor):
            return _try_copy(lhs, out)
    raise NotImplementedError()


@implements(torch.matmul)
def _matmul(lhs: Union[torch.Tensor, CustomTensor], rhs: Union[torch.Tensor, CustomTensor], out: Optional[torch.Tensor]=None) -> Union[torch.Tensor, CustomTensor]:
    if len(lhs.shape) != 2 or len(rhs.shape) != 2:
        raise ValueError('The arguments must be matrices.')
    if lhs.shape[1] != rhs.shape[0]:
        raise ValueError('Shape mismatch.')
    if isinstance(lhs, IdentityMatrix):
        return _try_copy(rhs, out)
    elif isinstance(lhs, ZeroTensor):
        return _try_copy(ZeroTensor((lhs.shape[0], rhs.shape[1]), dtype=lhs.dtype), out)
    elif isinstance(lhs, torch.Tensor):
        if isinstance(rhs, IdentityMatrix):
            return _try_copy(lhs, out)
        if isinstance(rhs, ZeroTensor):
            return _try_copy(ZeroTensor((lhs.shape[0], rhs.shape[1]), dtype=lhs.dtype), out)
    raise NotImplementedError()


@implements(torch.tensordot)
def _tensordot(lhs: Union[torch.Tensor, CustomTensor], rhs: Union[torch.Tensor, CustomTensor], dims: Union[int, Tuple[Iterable[int], Iterable[int]]]) -> Union[torch.Tensor, CustomTensor]:
    if isinstance(dims, int):
        if dims > len(lhs.shape) or dims > len(rhs.shape):
            raise ValueError('Dimension out of range.')
        if lhs.shape[-dims:] != rhs.shape[:dims]:
            raise ValueError('Contracted dimensions need to match.')
        return ZeroTensor((*lhs.shape[:-dims], *rhs.shape[dims:]), dtype=lhs.dtype)
    elif isinstance(dims, Tuple):
        if not all((lhs.shape[dim1] == rhs.shape[dim2] for dim1, dim2 in zip(*dims))):
            raise ValueError('Contracted dimensions need to match.')
        return ZeroTensor((*(size for dim, size in enumerate(lhs.shape) if not dim in dims[0]), *(size for dim, size in enumerate(rhs.shape) if not dim in dims[1])), dtype=lhs.dtype)
    raise NotImplementedError()


def _try_copy(src: Union[torch.Tensor, CustomTensor], dst: Union[torch.Tensor, CustomTensor]) -> Union[torch.Tensor, CustomTensor]:
    if dst is None:
        return src
    if isinstance(src, IdentityMatrix):
        if isinstance(dst, IdentityMatrix):
            if src.shape != dst.shape:
                raise ValueError('Shape mismatch.')
            return dst
        elif isinstance(dst, torch.Tensor):
            return dst.copy_(src.to_sparse())
    elif isinstance(src, ZeroTensor):
        if isinstance(dst, ZeroTensor):
            if src.shape != dst.shape:
                raise ValueError('Shape mismatch.')
            return dst
        elif isinstance(dst, torch.Tensor):
            return dst.copy_(src.to_sparse())
    elif isinstance(src, torch.Tensor):
        if isinstance(dst, torch.Tensor):
            return dst.copy_(src)
    raise NotImplementedError()
