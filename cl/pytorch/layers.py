from .activation import BaseActivation, ConformalActivation, NoActivation
from .convolution import BaseConvolution, Convolution, NoConvolution
from .dropout import BaseDropout, Dropout, NoDropout
from .module import ConformalModule
from .pooling import BasePooling, MeanPooling, NoPooling
from typing import Tuple, Union
import torch


class ConformalLayer:
    def __init__(self, convolution: BaseConvolution, activation: BaseActivation, dropout: BaseDropout, pooling: BasePooling):
        self.__convolution = convolution
        self.__activation = activation
        self.__dropout = dropout
        self.__pooling = pooling

    @property
    def activation(self) -> BaseActivation:
        return self.__activation

    @property
    def convolution(self) -> BaseConvolution:
        return self.__convolution

    @property
    def dropout(self) -> BaseDropout:
        return self.__dropout

    @property
    def pooling(self) -> BasePooling:
        return self.__pooling


class ConformalLayers(torch.nn.Module):
    def __init__(self):
        super(ConformalLayers, self).__init__()
        self.__layers = list()
        self.__valid_cache = False

    def __getitem__(self, index: int) -> ConformalLayer:
        return self.__layers[index]

    def __len__(self) -> int:
        return len(self.__layers)

    def __invalidate_cache(self) -> None:
        self.__valid_cache = False

    def __update_cache(self) -> None:
        if self.__valid_cache:
            return
        #TODO Do something
        self.__valid_cache = True
        raise NotImplementedError("To be implemented")

    def enqueue_layer(self, kernel_size: Union[int, Tuple[int, ...]], bias: bool=True, activation: bool=True, dropout_rate: float=None, pooling_size: Union[int, Tuple[int, ...]]=None) -> None:
        self.enqueue_module(Convolution(kernel_size=kernel_size, bias=bias))
        if activation:
            self.enqueue_module(ConformalActivation())
        if dropout_rate is not None:
            self.enqueue_module(Dropout(rate=dropout_rate))
        if pooling_size is not None:
            self.enqueue_module(MeanPooling(kernel_size=pooling_size))

    def enqueue_module(self, module: ConformalModule, name: str=None) -> None:
        self.__invalidate_cache()
        #TODO Do something
        raise NotImplementedError("To be implemented")

    def forward(self, input):
        self.__update_cache()
        #TODO Do something
        raise NotImplementedError("To be implemented")

    #TODO How to save?
    #TODO How to load?
    #TODO How to implement backward?
    #TODO How to call __invalidate_cache when the parameters change?
