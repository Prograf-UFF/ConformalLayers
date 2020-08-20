from .module import ConformalModule
from .utils import EyeTensor, _int_or_size_1_t, _int_or_size_2_t, _int_or_size_3_t, _size_any_t, _pair, _single, _triple
from abc import abstractmethod
from typing import Optional, Union
import torch


class BaseConv(ConformalModule):
    def __init__(self, name: Optional[str]=None) -> None:
        super(BaseConv, self).__init__(name)

    @property
    @abstractmethod
    def tensor(self) -> Union[torch.Tensor, EyeTensor]:
        pass


class ConvNd(BaseConv):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_any_t,
                 transposed: bool=False,
                 bias: bool=True,
                 name: Optional[str]=None) -> None:
        super(ConvNd, self).__init__(name)
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._transposed = transposed
        if transposed:
            self._weights = torch.nn.Parameter(torch.Tensor(in_channels, out_channels, *kernel_size))
        else:
            self._weights = torch.nn.Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        self._bias = torch.nn.Parameter(torch.Tensor(out_channels)) if bias else None

    def __repr__(self) -> str:
       return "Conv(in_channels={}, out_channels={}, kernel_size={}, transposed={}, bias={}{})".format(self._in_channels, self._out_channels, self._kernel_size, self._transposed, self._bias is not None, self._extra_repr(True))

    def _register_parent(self, parent, layer: int) -> None:
        parent.register_parameter("clayer[{}]-{}-weights".format(layer, "conv" if self._name is None else self._name), self._weights)
        self._weights.register_hook(lambda _: parent.invalidate_cache())
        if self._bias is not None:
            parent.register_parameter("clayer[{}]-{}-bias".format(layer, "conv" if self._name is None else self._name), self._bias)
            self._bias.register_hook(lambda _: parent.invalidate_cache())

    @property
    def bias(self) -> Optional[torch.nn.Parameter]:
        return self._bias

    @property
    def kernel_size(self) -> _size_any_t:
        return self._kernel_size

    @property
    def tensor(self) -> torch.Tensor:
        #TODO To be implemented
        raise NotImplementedError("To be implemented")

    @property
    def transposed(self) -> bool:
        return self._transposed

    @property
    def weights(self) -> torch.nn.Parameter:
        return self._weights


class Conv1d(ConvNd):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _int_or_size_1_t,
                 bias: bool=True,
                 name: Optional[str]=None) -> None:
        super(Conv1d, self).__init__(
            in_channels,
            out_channels,
            _single(kernel_size),
            False,
            bias,
            name)


class Conv2d(ConvNd):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _int_or_size_2_t,
                 bias: bool=True,
                 name: Optional[str]=None) -> None:
        super(Conv2d, self).__init__(
            in_channels,
            out_channels,
            _pair(kernel_size),
            False,
            bias,
            name)


class Conv3d(ConvNd):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _int_or_size_3_t,
                 bias: bool=True,
                 name: Optional[str]=None) -> None:
        super(Conv3d, self).__init__(
            in_channels,
            out_channels,
            _triple(kernel_size),
            False,
            bias,
            name)


class ConvTransposeNd(ConvNd):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_any_t,
                 bias: bool=True,
                 name: Optional[str]=None) -> None:
        super(ConvTransposeNd, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            True,
            bias,
            name)


class ConvTranspose1d(ConvNd):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _int_or_size_1_t,
                 bias: bool=True,
                 name: Optional[str]=None) -> None:
        super(ConvTranspose1d, self).__init__(
            in_channels,
            out_channels,
            _single(kernel_size),
            True,
            bias,
            name)


class ConvTranspose2d(ConvNd):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _int_or_size_2_t,
                 bias: bool=True,
                 name: Optional[str]=None) -> None:
        super(ConvTranspose2d, self).__init__(
            in_channels,
            out_channels,
            _pair(kernel_size),
            True,
            bias,
            name)


class ConvTranspose3d(ConvNd):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _int_or_size_3_t,
                 bias: bool=True,
                 name: Optional[str]=None) -> None:
        super(ConvTranspose3d, self).__init__(
            in_channels,
            out_channels,
            _triple(kernel_size),
            True,
            bias,
            name)


class NoConv(BaseConv):
    _instance = None

    def __init__(self):
        super(NoConv, self).__init__("NoConv")

    def __new__(cls) -> None:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
       return "NoConv()"

    @property
    def tensor(self) -> EyeTensor:
        return EyeTensor()
