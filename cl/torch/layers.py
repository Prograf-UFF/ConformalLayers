from .activation import BaseActivation, NoActivation
from .convolution import BaseConv, NoConv
from .dropout import BaseDropout, NoDropout
from .module import ConformalModule
from .pooling import BasePool, NoPool
from typing import Iterator
import torch


class ConformalLayer:
    def __init__(self,
                 conv: BaseConv=NoConv(),
                 activation: BaseActivation=NoActivation(),
                 dropout: BaseDropout=NoDropout(),
                 pool: BasePool=NoPool()) -> None:
        self._conv = conv
        self._activation = activation
        self._dropout = dropout
        self._pool = pool

    def __repr__(self) -> str:
        return "({}{}{}{})".format(
            "{}, ".format(self._conv) if not isinstance(self._conv, NoConv) else "",
            "{}, ".format(self._activation) if not isinstance(self._activation, NoActivation) else "",
            "{}, ".format(self._dropout) if not isinstance(self._dropout, NoDropout) else "",
            "{}, ".format(self._pool) if not isinstance(self._pool, NoPool) else "").strip()

    @property
    def activation(self) -> BaseActivation:
        return self._activation

    @property
    def conv(self) -> BaseConv:
        return self._conv

    @property
    def dropout(self) -> BaseDropout:
        return self._dropout

    @property
    def pool(self) -> BasePool:
        return self._pool


class ConformalLayers(torch.nn.Module):
    def __init__(self) -> None:
        super(ConformalLayers, self).__init__()
        self._layers = list()
        self._valid_cache = False
        self._cached_left_tensor = None
        self._cached_right_tensor = None

    def __repr__(self) -> str:
        r = "ConformalLayers("
        for ind, layer in enumerate(self._layers):
            if ind > 0:
                r += ", "
            r += "clayer[{}] = {}".format(ind, layer)
        r += ")"
        return r

    def __getitem__(self, index: int) -> ConformalLayer:
        return self._layers[index]

    def __iter__(self) -> Iterator[ConformalLayer]:
        return iter(self._layers)
    
    def __len__(self) -> int:
        return len(self._layers)

    def _update_cache(self) -> None:
        if self._valid_cache:
            return
        #TODO Do something
        self._valid_cache = True
        raise NotImplementedError("To be implemented")

    def enqueue_module(self, module: ConformalModule) -> None:
        if isinstance(module, (NoConv, NoActivation, NoDropout, NoPool)):
            return
        if isinstance(module, BaseConv):
            self._layers.append(ConformalLayer(conv=module))
        elif isinstance(module, BaseActivation):
            if not self._layers or not isinstance(self._layers[-1].activation, NoActivation) or not isinstance(self._layers[-1].dropout, NoDropout) or not isinstance(self._layers[-1].pool, NoPool):
                self._layers.append(ConformalLayer(activation=module))
            else:
                self._layers[-1]._activation = module
        elif isinstance(module, BaseDropout):
            if not self._layers or not isinstance(self._layers[-1].dropout, NoDropout) or not isinstance(self._layers[-1].pool, NoPool):
                self._layers.append(ConformalLayer(dropout=module))
            else:
                self._layers[-1]._dropout = module
        elif isinstance(module, BasePool):
            if not self._layers or not isinstance(self._layers[-1].pool, NoPool):
                self._layers.append(ConformalLayer(pool=module))
            else:
                self._layers[-1]._pool = module
        else:
            raise NotImplementedError("Invalid module: {}".format(module))
        self.invalidate_cache()
        module._register_parent(self, len(self._layers)-1)

    def enqueue_modules(self, module: ConformalModule, *args: ConformalModule) -> None:
        self.enqueue_module(module)
        for module in args:
            self.enqueue_module(module)

    def forward(self, x):
        self._update_cache()
        #TODO Inclusão da coordenada homogênea em x
        y = torch.matmul(torch.add(self._cached_left_tensor, torch.matmul(self._cached_right_tensor, x)), x)
        #TODO Normalização da coordenada homogênea e remoção da coordenada homogênea de y
        return y

    def invalidate_cache(self) -> None:
        self._valid_cache = False

    @property
    def valid_cache(self) -> bool:
        return self._valid_cache
