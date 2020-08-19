from .activation import BaseActivation, ConformalActivation, NoActivation
from .convolution import BaseConvolution, Convolution, NoConvolution
from .dropout import BaseDropout, Dropout, NoDropout
from .module import ConformalModule
from .pooling import BasePooling, AveragePooling, NoPooling
from typing import List, Optional, Tuple, Union
import torch


class ConformalLayer:
    def __init__(self, convolution: Optional[BaseConvolution]=NoConvolution(), activation: Optional[BaseActivation]=NoActivation(), dropout: Optional[BaseDropout]=NoDropout(), pooling: Optional[BasePooling]=NoPooling()):
        self._convolution = convolution
        self._activation = activation
        self._dropout = dropout
        self._pooling = pooling

    def __repr__(self) -> str:
        return "ConformalLayer({})".format("{}{}{}{}".format(
            "{}, ".format(self._convolution) if not isinstance(self._convolution, NoConvolution) else "",
            "{}, ".format(self._activation) if not isinstance(self._activation, NoActivation) else "",
            "{}, ".format(self._dropout) if not isinstance(self._dropout, NoDropout) else "",
            "{}, ".format(self._pooling) if not isinstance(self._pooling, NoPooling) else "").strip())

    @property
    def activation(self) -> BaseActivation:
        return self._activation

    @property
    def convolution(self) -> BaseConvolution:
        return self._convolution

    @property
    def dropout(self) -> BaseDropout:
        return self._dropout

    @property
    def pooling(self) -> BasePooling:
        return self._pooling


class ConformalLayers(torch.nn.Module):
    def __init__(self):
        super(ConformalLayers, self).__init__()
        self._layers = list()
        self._valid_cache = False
        self._cached_left_tensor = None
        self._cached_right_tensor = None

    def __repr__(self) -> str:
       return "ConformalLayers(layers={})".format(self._layers)

    def __getitem__(self, index: int) -> ConformalLayer:
        return self._layers[index]

    def __len__(self) -> int:
        return len(self._layers)

    def _invalidate_cache(self) -> None:
        self._valid_cache = False

    def _update_cache(self) -> None:
        if self._valid_cache:
            return
        #TODO Do something
        self._valid_cache = True
        raise NotImplementedError("To be implemented")

    def enqueue_layer(self, kernel_size: Union[int, Tuple[int, ...]], bias: Optional[bool]=True, activation: Optional[bool]=True, dropout_rate: Optional[float]=None, pooling_size: Optional[Union[int, Tuple[int, ...]]]=None) -> None:
        self.enqueue_module(Convolution(kernel_size=kernel_size, bias=bias))
        if activation:
            self.enqueue_module(ConformalActivation())
        if dropout_rate is not None:
            self.enqueue_module(Dropout(rate=dropout_rate))
        if pooling_size is not None:
            self.enqueue_module(AveragePooling(kernel_size=pooling_size))

    def enqueue_module(self, module: ConformalModule) -> None:
        if isinstance(module, (NoConvolution, NoActivation, NoDropout, NoPooling)):
            return
        if isinstance(module, BaseConvolution):
            self._layers.append(ConformalLayer(convolution=module))
        elif isinstance(module, BaseActivation):
            if not self._layers or not isinstance(self._layers[-1].activation, NoActivation) or not isinstance(self._layers[-1].dropout, NoDropout) or not isinstance(self._layers[-1].pooling, NoPooling):
                self._layers.append(ConformalLayer(activation=module))
            else:
                self._layers[-1]._activation = module
        elif isinstance(module, BaseDropout):
            if not self._layers or not isinstance(self._layers[-1].dropout, NoDropout) or not isinstance(self._layers[-1].pooling, NoPooling):
                self._layers.append(ConformalLayer(dropout=module))
            else:
                self._layers[-1]._dropout = module
        elif isinstance(module, BasePooling):
            if not self._layers or not isinstance(self._layers[-1].pooling, NoPooling):
                self._layers.append(ConformalLayer(pooling=module))
            else:
                self._layers[-1]._pooling = module
        else:
            raise NotImplementedError("Invalid module: {}".format(module))
        self._invalidate_cache()

    def enqueue_modules(self, *args: ConformalModule) -> None:
        for arg in args:
            self.enqueue_module(arg)

    def forward(self, x):
        self._update_cache()
        return torch.matmul(torch.add(self._cached_left_tensor, torch.matmul(self._cached_right_tensor, x)), x)

    #TODO How to save?
    #TODO How to load?
    #TODO How to implement backward?
    #TODO How to call _invalidate_cache when the parameters change?
