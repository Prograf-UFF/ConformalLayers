from .layers import ConformalLayers
from .utils import DenseTensor
from typing import Iterator, List
import torch


class SGD(torch.optim.SGD):
    def __init__(self, params: Iterator[DenseTensor], clayers: Iterator[ConformalLayers], *args, **kwargs) -> None:
        super(SGD, self).__init__(params, *args, **kwargs)
        self.param_groups[0]['clayers'] = list(clayers)  # We need a new entry to keep the ConformalLayer objects

    @torch.no_grad()
    def step(self, closure=None):
        # The following code is a copy of the SGD.step() implementation with few modifications
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for clayers in group['clayers']:  # Instead of update the tensors in group['params'], we update the cached tensors of each ConformalLayer object in group['clayers']
                for p in clayers.cached_data():
                    if p.grad is None:
                        continue
                    d_p = p.grad
                    if weight_decay != 0:
                        d_p = d_p.add(p, alpha=weight_decay)
                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                        if nesterov:
                            d_p = d_p.add(buf, alpha=momentum)
                        else:
                            d_p = buf

                    p.data.add_(d_p, alpha=-group['lr'])  # We call p.data.add_() instead of p.add_() to preserve the graph used to compute the grads of cached tensors

        # The original implementation of SGD.step() is called here
        loss = super(SGD, self).step(closure)
        return loss
