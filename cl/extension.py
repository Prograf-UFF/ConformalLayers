from .decorator import singleton
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Callable, Iterable, Optional, Union
import functools, numpy, re, torch, torch_sparse

assert torch_sparse.__version__ == '0.4.4', 'Proper autograd support in torch_sparse.spspmm() implemented only in version 0.4.4 (see https://github.com/rusty1s/pytorch_sparse/issues/45).'


HANDLED_FUNCTIONS = {}


def implements(torch_function: Callable):
    @functools.wraps(torch_function)
    def decorator(func):
        HANDLED_FUNCTIONS[torch_function] = func
        return func
    return decorator


class CustomTensor(ABC):
    def __init__(self, size: Iterable[int]) -> None:
        super(CustomTensor, self).__init__()
        self._size = torch.Size(size)

    def __repr__(self) -> str:
        entries_str = ',\n    '.join(map(lambda pair: '{}={}'.format(*pair), self._repr_dict().items()))
        return f'{self.__class__.__name__}(\n    {entries_str})'

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def _repr_dict(self) -> OrderedDict:
        entries = OrderedDict()
        entries['size'] = tuple(map(int, self.shape))
        entries['nelement'] = self.nelement
        entries['nnz'] = self.nnz
        entries['dtype'] = self.dtype
        return entries

    def add(self, other: Union[torch.Tensor, 'CustomTensor']) -> Union[torch.Tensor, 'CustomTensor']:
        return torch.add(self, other)

    @abstractmethod
    def copy_(self, scr: 'CustomTensor') -> 'CustomTensor':
        pass
    
    def dim(self) -> int:
        return len(self._size)

    def matmul(self, other: Union[torch.Tensor, 'CustomTensor']) -> Union[torch.Tensor, 'CustomTensor']:
        return torch.matmul(self, other)
    
    def mm(self, other: Union[torch.Tensor, 'CustomTensor']) -> Union[torch.Tensor, 'CustomTensor']:
        return torch.mm(self, other)

    def numel(self) -> int:
        return numpy.prod(self.shape)
    
    @abstractmethod
    def permute(self, *dims: int) -> 'CustomTensor':
        pass

    def size(self) -> torch.Size:
        return self._size

    @abstractmethod
    def t(self) -> 'CustomTensor':
        pass
    
    @abstractmethod
    def to_native(self) -> torch.Tensor:
        pass

    @property
    def ndim(self) -> int:
        return len(self._size)

    @property
    @abstractmethod
    def nnz(self) -> int:
        pass

    @property
    def nelement(self) -> int:
        return numpy.prod(self.shape)

    @property
    @abstractmethod
    def device(self) -> torch.device:
        pass

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        pass

    @property
    def shape(self) -> torch.Size:
        return self._size


class IdentityMatrix(CustomTensor):
    def __init__(self, size: int, dtype: torch.dtype, device: torch.device) -> None:
        super(IdentityMatrix, self).__init__((size, size))
        self._dtype = dtype
        self._device = device

    def copy_(self, src: 'IdentityMatrix') -> 'IdentityMatrix':
        if isinstance(scr, IdentityMatrix) and self.shape == scr.shape:
            return self
        raise RuntimeError('copy_() expects a src IdentityMatrix object with the same size as self.')

    def permute(self, *dims: int) -> 'IdentityMatrix':
        dims = numpy.asarray(dims)
        if self.ndim != len(dims):
            raise ValueError('Number of dims don''t match in permute.')
        if self.ndim != len(numpy.unique(dims)):
            raise ValueError('Repeated dim in permute.')
        if numpy.any(dims < -self.ndim) or numpy.any(self.ndim <= dims):
            raise ValueError('Dimension out of range.')
        return self

    def t(self) -> 'IdentityMatrix':
        return self

    def to_native(self) -> torch.Tensor:
        ind = numpy.arange(self.shape[0], dtype=numpy.int64, device=self.device)
        return torch.sparse_coo_tensor((ind, ind), torch.ones((self.shape[0],), dtype=self.dtype, device=self.device), size=self.shape)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def nnz(self) -> int:
        return self.shape[0]


class SparseTensor(CustomTensor):
    def __init__(self, indices: torch.LongTensor, values: torch.Tensor, size: Iterable[int], *, coalesced: bool) -> None:
        super(SparseTensor, self).__init__(size)
        self._indices = torch.as_tensor(indices, dtype=torch.int64, device=values.device)
        self._values = torch.as_tensor(values, device=values.device)
        self._coalesced = coalesced

    def _repr_dict(self) -> OrderedDict:
        entries = super()._repr_dict()
        entries['indices'] = [*map(lambda arg: tuple(map(int, arg)), self.indices.t())]
        entries['values'] = re.sub('(?:\n *)', ' ', repr(self.values))
        return entries

    def coalesce(self, force: Optional[bool]=False):
        if force or not self.coalesced:
            if self.ndim != 2:
                raise NotImplementedError()
            self._indices, self._values = torch_sparse.coalesce(self._indices, self._values, self.shape[0], self.shape[1])
            self._coalesced = True

    def copy_(self, src: CustomTensor) -> 'SparseTensor':
        if isinstance(scr, IdentityMatrix): #TODO Lidar com broadcast
            raise NotImplementedError() #TODO Implementar
        elif isinstance(scr, SparseTensor):
            raise NotImplementedError() #TODO Implementar
        elif isinstance(scr, ZeroTensor):
            raise NotImplementedError() #TODO Implementar
        raise RuntimeError('copy_() expects a src CustomTensor object broadcastable with self.')

    def permute(self, *dims: int) -> 'SparseTensor':
        dims = numpy.asarray(dims)
        if self.ndim != len(dims):
            raise ValueError('Number of dims don''t match in permute.')
        if self.ndim != len(numpy.unique(dims)):
            raise ValueError('Repeated dim in permute.')
        if numpy.any(dims < -self.ndim) or numpy.any(self.ndim <= dims):
            raise ValueError('Dimension out of range.')
        return SparseTensor(self.indices[dims], self.values, (self.shape[dim] for dim in dims), coalesced=False)

    def t(self) -> 'SparseTensor':
        if self.ndim == 2:
            return SparseTensor(self.indices[[1, 0]], self.values, (self.shape[1], self.shape[0]), coalesced=False)
        elif self.ndim < 2:
            return self
        raise RuntimeError('t() expects a tensor with <= 2 dimensions.')

    def to_native(self) -> torch.Tensor:
        return torch.sparse_coo_tensor(self._indices, self._values, size=self.shape)

    @property
    def coalesced(self) -> bool:
        return self._coalesced

    @property
    def device(self) -> torch.device:
        return self._values.device

    @property
    def dtype(self) -> torch.dtype:
        return self._values.dtype

    @property
    def indices(self) -> torch.LongTensor:
        return self._indices

    @property
    def nnz(self) -> int:
        self.coalesce()
        return len(self.values)

    @property
    def values(self) -> torch.Tensor:
        return self._values


class ZeroTensor(CustomTensor):
    def __init__(self, size: Iterable[int], dtype: torch.dtype, device: torch.device) -> None:
        super(ZeroTensor, self).__init__(size)
        self._dtype = dtype
        self._device = device

    def copy_(self, scr: 'ZeroTensor') -> 'ZeroTensor':
        if isinstance(scr, ZeroTensor): #TODO Lidar com broadcast
            return self
        raise RuntimeError('copy_() expects a src ZeroTensor object broadcastable with self.')

    def permute(self, *dims: int) -> 'ZeroTensor':
        dims = numpy.asarray(dims)
        if self.ndim != len(dims):
            raise ValueError('Number of dims don''t match in permute.')
        if self.ndim != len(numpy.unique(dims)):
            raise ValueError('Repeated dim in permute.')
        if numpy.any(dims < -self.ndim) or numpy.any(self.ndim <= dims):
            raise ValueError('Dimension out of range.')
        return ZeroTensor((self.shape[dim] for dim in dims), dtype=self.dtype, device=self.device)

    def t(self) -> 'ZeroTensor':
        if self.ndim == 2:
            return ZeroTensor((self.shape[1], self.shape[0]), dtype=self.dtype, device=self.device)
        elif self.ndim < 2:
            return self
        raise RuntimeError('t() expects a tensor with <= 2 dimensions.')

    def to_native(self) -> torch.Tensor:
        return torch.sparse_coo_tensor(size=self.shape, dtype=self.dtype, device=self.device)

    def view(self, *shape: int) -> 'ZeroTensor':
        view_shape = numpy.asarray(shape)
        if numpy.any(view_shape < -1):
            raise ValueError('Invalid shape dimension.')
        inferred = view_shape == -1
        ninferred = numpy.count_nonzero(inferred)
        size = numpy.prod(self.shape)
        if ninferred == 1:
            view_shape[inferred] = size // numpy.prod(view_shape[numpy.logical_not(inferred)])
        elif ninferred > 1:
            raise ValueError('Only one dimension can be inferred.')
        if size != numpy.prod(view_shape):
            raise ValueError(f'Shape {*map(int, shape),} is invalid for input of size {size}.') 
        return ZeroTensor(view_shape, dtype=self.dtype, device=self.device)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def nnz(self) -> int:
        return 0


@implements(torch.add)
def _add(lhs: Union[torch.Tensor, CustomTensor], rhs: Union[torch.Tensor, CustomTensor], *, out: Optional[Union[torch.Tensor, CustomTensor]]=None) -> Union[torch.Tensor, CustomTensor]:
    lsize, rsize = lhs.shape, rhs.shape
    if (lhs.ndim <= rhs.ndim and lsize != rsize[-lhs.ndim:]) or (lhs.ndim > rhs.ndim and lsize[-rhs.ndim:] != rsize):
        raise ValueError('Shape mismatch.')
    if isinstance(lhs, IdentityMatrix):
        if isinstance(rhs, IdentityMatrix):
            ind = numpy.arange(lsize[0], dtype=numpy.int64)
            return _try_copy(SparseTensor((ind, ind), torch.fill((lsize[0],), 2, dtype=lhs.dtype, device=lhs.device), lsize, coalesced=True))
        elif isinstance(rhs, SparseTensor):
            ind = numpy.arange(lsize[0], dtype=numpy.int64)
            return _try_copy(SparseTensor(torch.cat((rhs.indices, torch.as_tensor((ind, ind), dtype=torch.int64, device=rhs.device)), 1), torch.cat((rhs.values, torch.ones((lsize[0],), dtype=rhs.dtype, device=rhs.device)), 0), lsize, coalesced=False), out) #TODO Lidar com broadcast
        elif isinstance(rhs, ZeroTensor):
            return _try_copy(lhs, out) #TODO Lidar com broadcast
        elif isinstance(rhs, torch.Tensor):
            return torch.add(rhs, lhs.to_native(), out=out)
    elif isinstance(lhs, SparseTensor):
        if isinstance(rhs, IdentityMatrix):
            ind = numpy.arange(rsize[0], dtype=numpy.int64)
            return _try_copy(SparseTensor(torch.cat((lhs.indices, torch.as_tensor((ind, ind), dtype=torch.int64, device=lhs.device)), 1), torch.cat((lhs.values, torch.ones((rsize[0],), dtype=lhs.dtype, device=lhs.device)), 0), lsize, coalesced=False), out) #TODO Lidar com broadcast
        elif isinstance(rhs, SparseTensor):
            return _try_copy(SparseTensor(torch.cat((lhs.indices, rhs.indices), 1), torch.cat((lhs.values, rhs.values), 0), lsize, coalesced=False), out) #TODO Lidar com broadcast
        elif isinstance(rhs, ZeroTensor):
            return _try_copy(lhs, out) #TODO Lidar com broadcast
        elif isinstance(rhs, torch.Tensor):
            return torch.add(rhs, lhs.to_native(), out=out)
    elif isinstance(lhs, ZeroTensor):
        return _try_copy(rhs, out) #TODO Lidar com broadcast
    elif isinstance(lhs, torch.Tensor):
        if isinstance(rhs, (IdentityMatrix, SparseTensor)):
            return torch.add(lhs, rhs.to_native(), out=out)
        elif isinstance(rhs, ZeroTensor):
            return _try_copy(lhs, out) #TODO Lidar com broadcast
    raise NotImplementedError()


@implements(torch.matmul)
def _matmul(lhs: Union[torch.Tensor, CustomTensor], rhs: Union[torch.Tensor, CustomTensor], *, out: Optional[Union[torch.Tensor, CustomTensor]]=None) -> Union[torch.Tensor, CustomTensor]:
    def out_size():
        lsize, rsize = lhs.shape, rhs.shape
        if lhs.ndim == 1 and rhs.ndim == 1 and lsize[-1] == rsize[-1]: return (1,)
        if lhs.ndim == 2 and rhs.ndim == 2 and lsize[-1] == rsize[-2]: return (lsize[0], rsize[1])
        if lhs.ndim == 1 and rhs.ndim == 2 and lsize[-1] == rsize[-2]: return (rsize[1],)
        if lhs.ndim == 2 and rhs.ndim == 1 and lsize[-1] == rsize[-1]: return (lsize[0],)
        if (lhs.ndim >= 1 and rhs.ndim >= 2) or (lhs.ndim >= 2 and rhs.ndim >= 1):
            if lhs.ndim == 1 and lsize[-1] == rsize[-2]: return (*numpy.maximum(1, rsize[:-2]), *rsize[-1:])
            if rhs.ndim == 1 and lsize[-1] == rsize[-1]: return (*numpy.maximum(lsize[:-2], 1), *lsize[-2:-1])
            if lsize[-1] == rsize[-2]: 
                if lhs.ndim <= rhs.ndim: return (*numpy.maximum(1, rsize[:-lhs.ndim]), *numpy.maximum(lsize[:-2], rsize[-lhs.ndim:-2]), lsize[-2], rsize[-1])
                else: return (*numpy.maximum(lsize[:-rhs.ndim], 1), *numpy.maximum(lsize[-rhs.ndim:-2], rsize[:-2]), lsize[-2], rsize[-1])
        raise ValueError('Shape mismatch.')
    if isinstance(lhs, IdentityMatrix):
        return _try_copy(rhs, out)
    elif isinstance(lhs, SparseTensor):
        if isinstance(rhs, IdentityMatrix):
            return _try_copy(lhs, out)
        elif isinstance(rhs, SparseTensor):
            lhs.coalasce()
            rhs.coalasce()
            if lhs.ndim == 1:
                lsize = (1, lhs.shape[-1])
                lindices = (numpy.zeros((len(lhs.indices),), dtype=numpy.int64), lhs.indices[0])
            elif lhs.ndim == 2:
                lsize = lhs.shape
                lindices = lhs.indices
            else:
                raise NotImplementedError() #TODO Lidar com broadcast
            if rhs.ndim == 1:
                rsize = (rhs.shape[-1], 1)
                rindices = (rhs.indices[0], numpy.zeros((len(rhs.indices),), dtype=numpy.int64))
            elif rhs.ndim == 2:
                rsize = rhs.shape
                rindices = rhs.indices
            else:
                raise NotImplementedError() #TODO Lidar com broadcast
            indices, values = torch_sparse.spspmm(lindices, lhs.values, rindices, rhs.values, lsize[0], lsize[1], rsize[1])
            if lhs.ndim == 1:
                return _try_copy(SparseTensor(indices[1], values, (rsize[1],), coalesced=True), out)
            elif rhs.ndim == 1:
                return _try_copy(SparseTensor(indices[0], values, (lsize[0],), coalesced=True), out)
            else:
                return _try_copy(SparseTensor(indices, values, (lsize[0], rsize[1]), coalesced=True), out)
        elif isinstance(rhs, ZeroTensor):
            return _try_copy(ZeroTensor(out_size(), dtype=lhs.dtype, device=lhs.device), out)
        elif isinstance(rhs, torch.Tensor):
            return torch.matmul(lhs.to_native(), rhs, out=out)  #TODO autograd implementado?
    elif isinstance(lhs, ZeroTensor):
        return _try_copy(ZeroTensor(out_size(), dtype=lhs.dtype, device=lhs.device), out)
    elif isinstance(lhs, torch.Tensor):
        if isinstance(rhs, IdentityMatrix):
            return _try_copy(lhs, out)
        elif isinstance(rhs, SparseTensor):
            return torch.matmul(lhs, rhs.to_native(), out=out) #TODO autograd implementado?
        elif isinstance(rhs, ZeroTensor):
            return _try_copy(ZeroTensor(out_size(), dtype=lhs.dtype, device=lhs.device), out)
    raise NotImplementedError()


@implements(torch.mm)
def _mm(lhs: Union[torch.Tensor, CustomTensor], rhs: Union[torch.Tensor, CustomTensor]) -> Union[torch.Tensor, CustomTensor]:
    if lhs.ndim != 2 or rhs.ndim != 2:
        raise ValueError('Matrices expected.')
    if lhs.shape[1] != rhs.shape[0]:
        raise ValueError('Shape mismatch.')
    if isinstance(lhs, IdentityMatrix):
        return rhs
    elif isinstance(lhs, SparseTensor):
        if isinstance(rhs, IdentityMatrix):
            return lhs
        elif isinstance(rhs, SparseTensor):
            lhs.coalesce()
            rhs.coalesce()
            indices, values = torch_sparse.spspmm(lhs.indices, lhs.values, rhs.indices, rhs.values, lhs.shape[0], lhs.shape[1], rhs.shape[1])
            return SparseTensor(indices, values, (lhs.shape[0], rhs.shape[1]), coalesced=True)
        elif isinstance(rhs, ZeroTensor):
            return ZeroTensor((lhs.shape[0], rhs.shape[1]), dtype=lhs.dtype, device=lhs.dtype)
        elif isinstance(rhs, torch.Tensor):
            lhs.coalesce()
            return torch_sparse.spmm(lhs.indices, lhs.values, lhs.shape[0], lhs.shape[1], rhs)
    elif isinstance(lhs, ZeroTensor):
        return ZeroTensor((lhs.shape[0], rhs.shape[1]), dtype=rhs.dtype, device=rhs.device)
    elif isinstance(lhs, torch.Tensor):
        if isinstance(rhs, IdentityMatrix):
            return lhs
        elif isinstance(rhs, SparseTensor):
            return torch.matmul(lhs, rhs.to_native()) #TODO autograd implementado?
        elif isinstance(rhs, ZeroTensor):
            return ZeroTensor((lhs.shape[0], rhs.shape[1]), dtype=lhs.dtype, device=lhs.device)
    raise NotImplementedError()


def _try_copy(src: Union[torch.Tensor, CustomTensor], dst: Union[torch.Tensor, CustomTensor]) -> Union[torch.Tensor, CustomTensor]:
    if dst is None:
        return src
    if isinstance(src, CustomTensor):
        if isinstance(dst, CustomTensor):
            return dst.copy_(src)
        elif isinstance(dst, torch.Tensor):
            return dst.copy_(src.to_native())
    elif isinstance(src, torch.Tensor):
        if isinstance(dst, torch.Tensor):
            return dst.copy_(src)
    raise NotImplementedError()
