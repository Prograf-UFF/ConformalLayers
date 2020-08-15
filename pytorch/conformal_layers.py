import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

class ConformalLayers(nn.Module):

    def __init__(self):
        super(ConformalLayers, self).__init__()

    def _invalidate(self):
        '''
        Método chamado sempre que _enqueue_layer(...) for ativado e
        quando nn.Parameters forem atualizados
        '''
        pass

    def _enqueue_layer(self, kernel_size=None, bias=False, activation=False, pooling=None, dropout=None, kwargs=dict()):
        '''
        Retorna o objeto de layer
        '''
        pass

    def enqueue_layer(self, kernel_size, bias=True, activation=True, pooling=None, dropout=None, kwargs=dict()):
        '''
        Retorna o objeto de layer criado
        '''
        pass

    def enqueue_convolution(self, kernel_size, bias=True, label=None):
        '''
        Retorna o objeto de convolution criado na última layer
        '''
        pass

    def enqueue_activation(self, label=None):
        '''
        Retorna o objeto de activation criado na última layer
        '''
        pass

    def enqueue_dropout(self, rate, label=None):
        '''
        Retorna o objeto de pooling criado na última layer
        '''
        pass

    def enqueue_pooling(self, window_size, label=None):
        '''
        Retorna o objeto de pooling criado na última layer
        '''
        pass

    def __len__(self):
        pass

    def getitem(self, k):
        pass

    def forward(self, ...):
        pass

    def backward(self, ...):
        pass

    def save(self, ...):
        pass

    def load(self, ...):
        pass
