# ConformalLayers: A non-linear sequential neural network with associative layers

ConformalLayers is a conformal embedding of sequential layers of Convolutional Neural Networks (CNNs) that allows associativity between operations like convolution, average pooling, dropout, flattening, padding, dilation, and stride. Such embedding allows associativity between layers of CNNs, considerably reducing the amount of operations to perform inference in type of neural networks. 

This repository is a implementation of ConformalLayers written in Python using [Minkowski Engine](https://github.com/NVIDIA/MinkowskiEngine) and [PyTorch](https://pytorch.org/) as backend. This implementation is a first step into the usage of activation functions, like ReSPro, that can be represented as tensors, depending on the geometry model.

Please cite our [SIBGRAPI'21](http://arxiv.org/abs/2110.12108) paper if you use this code in your research. The paper presents a complete description of the library:

```{.bib}
@InProceedings{sousa_et_al-sibgrapi-2021,
  author    = {Sousa, Eduardo V. and Fernandes, Leandro A. F. and Vasconcelos, Cristina N.},
  title     = {{C}onformal{L}ayers: a non-linear sequential neural network with associative layers},
  booktitle = {Proceedings of the 2021 34th SIBGRAPI Conference on Graphics, Patterns and Images},
  year      = {2021},
}
```

Please, let Eduardo Vera Sousa ([http://www.ic.uff.br/~eduardovera](http://www.ic.uff.br/~eduardovera)), Leandro A. F. Fernandes ([http://www.ic.uff.br/~laffernandes](http://www.ic.uff.br/~laffernandes)) and Cristina Nader Vasconcelos(http://www2.ic.uff.br/~crisnv/index.php) know if you want to contribute to this project. Also, do not hesitate to contact them if you encounter any problems.

**Contents:**

1. [Requirements](#1-requirements)
2. [How to Install ConformalLayers](#2-how-to-install-conformallayers)
3. [Running Examples](#3-running-examples)
4. [Compiling and Running Unit Tests](#4-running-unit-tests)
5. [Documentation](#5-documentation)
6. [License](#6-license)

## 1. Requirements

Make sure that you have the following tools before attempting to use ConformalLayers.

Required tools:

- [Python 3](https://www.python.org) interpreter

- [PyTorch](https://pytorch.org/) (version >= 1.8)

- [Minkowski Engine](https://github.com/NVIDIA/MinkowskiEngine/tree/v0.4.3) (version 0.4.3)

Optional tool to use ConformalLayers:

- [Virtual enviroment](https://wiki.archlinux.org/index.php/Python/Virtual_environment) to create an isolated workspace for a Python application.

- [Docker](https://www.docker.com/) to create a container to run ConformalLayers

## 2. How to Install ConformalLayers

No magic needed here. Just run:

```bash
python setup.py install
```
## 3. Running Examples

The basic steps for running the examples of ConformalLayers look like this:

```bash
cd <ConformalLayers-dir>/Experiments/<experiment-name>
```

For Experiments I and II, each file refers to the experiment described on the main paper. Thus, in order to run BaseReSProNet with FashionMNIST dataset, for example, all you have to do is:

```bash
python BaseReSProNet.py --dataset=FashionMNIST
```

The values that can be used for the ```dataset``` argument are 

- MNIST
- FashionMNIST
- CIFAR10

The loader of each dataset is described in ```Experiments/utils/datasets.py``` file. 

Other arguments for the script files in Experiments I and II are:
- epochs (```int``` value)
- batch_size (```int``` value)
- learning_rate (```float``` value)
- optimizer (```adam``` or ```rmsprop```)
- dropout (```float``` value)

For Experiments III and IV, since we compute the amount of memory used, we used an external file to orchestrate the calls and make sure we have a clean environment for the next iterations. Such orchestrator is writen on the files with the suffix ```_manager.py```.

You can also run the files that corresponds to each architecture individually, without the orchestrator. To run ```D3ModNetCL``` architecture, for example, just run

```bash
python D3ModNetCL.py
```

The arguments for the non-orchestrated scripts in Experiments III and IV are:
- num_inferences (```int``` value)
- batch_size (```int``` value)
- depth (```int``` value, Experiment III only)


The files in ```networks``` folder contains the description of each architecture used in our experiments and presents the usage of the classes and methods of our library.


## 4. Running Unit Tests

The basic steps for running the unit tests of ConformalLayers look like this:

```bash
cd <ConformalLayers-dir>/tests
```

To run all tests, simply run

```bash
python test_all.py
```

To run the tests for each module, run:

```bash
python test_<module_name>.py
```

## 5. Documentation

Here you find a brief description of the namespaces, macros, classes, functions, procedures, and operators available for the user. All methods are available with C++ and most of them with Python. The detailed documentation is not ready yet.

Contents:

- [Modules](#modules)
  - [Convolution](#convolution)
  - [Pooling](#pooling)
  - [Activation](#activation)
  - [Regularization](#regularization)

### Modules

Here we present the main modules implemented in our framework. Most of the modules are used just like in PyTorch, so users with some background on this framework benefits from this implementation. For users not familiar with PyTorch, the usage still quite simple and intuitive.  

| Module | Description |
| --- | --- |
| `Conv1d`, `Conv2d`, `Conv3d`, `ConvNd` | Convolution operation implemented for *n*-D signals |
| `AvgPool1d`, `AvgPool2d`, `AvgPool3d`, `AvgPoolNd` | Average pooling operation implemented for *n*-D signals |
| `BaseActivation` | The abstract class for the activation function layer. To extend the library, one shall implement this class  |
| `ReSPro` | The layer that corresponds to the ```ReSPro``` activation function. Such function is a linear function with non-linear behavior that can be encoded as a tensor. The non-linearity of this function is controlled by a parameter <span>&alpha;</span> (passed as argument) that can be provided or inferred from the data |
| `Regularization` | In this version, ```Dropout``` is only regularization available. In this approach, during the training phase, we randomly shut down some neurons with a probability ```p```, passed as argument to this module |
<br>

These modules are composed into ConformalLayers in a very similar way to the pure PyTorch-based way. The class ```ConformalLayers``` plays an important role in this task, as you can see by comparing the code snippets below:

```python
# This one is built with pure PyTorch
import torch.nn as nn

class D3ModNet(nn.Module):
    def __init__(self):
        super(D3ModNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.fc1 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        return x

```

```python
# This one is built with ConformalLayers
import torch.nn as nn
import ConformalLayers as cl

class D3ModNetCL(nn.Module):
    def __init__(self):
        super(D3ModNetCL, self).__init__()
        self.features = cl.ConformalLayers(
            cl.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
            cl.ReSPro(),
            cl.AvgPool2d(kernel_size=2, stride=2),
            cl.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
            cl.ReSPro(),
            cl.AvgPool2d(kernel_size=2, stride=2),
            cl.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
            cl.ReSPro(),
            cl.AvgPool2d(kernel_size=2, stride=2),
        )
        self.fc1 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        return x
```

They look pretty much the same code, right? That's because we've implemented ```ConformalLayers``` to be a transition smoothest as possible to the PyTorch user. Most of the modules has almost the same method signatures of the ones provided by PyTorch. 

#### <u>Convolution</u>

The convolution operation implemented in ConformalLayers on the modules ```ConvNd```, ```Conv1d```, ```Conv2d``` and ```Conv3d``` is almost the same one implemented on PyTorch but we do not allow bias. This is mostly due to the construction of our logic when building the representation with tensors. Although we have a few ideas on how to include bias on this representation, they are not included in the current version. The parameters are detailed below and are originally available in [PyTorch convolution documentation page](https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html#torch.nn.Conv1d). The exception here relies on the ```padding_mode``` parameter, that is always set to '```zeros```' in our implementation.


- in_channels (```int```) – Number of channels in the input image

- out_channels (```int```) – Number of channels produced by the convolution

- kernel_size (```int``` or ```tuple```) – Size of the convolving kernel

- stride (```int``` or ```tuple```, optional) – Stride of the convolution. Default: 1

- padding (```int```, ```tuple``` or ```str```, optional) – Padding added to both sides of the input. Default: 0

- dilation (```int``` or ```tuple```, optional) – Spacing between kernel elements. Default: 1

- groups (```int```, optional) – Number of blocked connections from input channels to output channels. Default: 1


#### <u>Pooling</u>

In our current implementation, we only support average pooling, which is implemented on modules ```AvgPoolNd```, ```AvgPool1d```, ```AvgPool2d``` and ```AvgPool3d```. The parameters list, originally available in [PyTorch average pooling documentation page](https://pytorch.org/docs/stable/generated/torch.nn.AvgPool1d.html#torch.nn.AvgPool1d), is described below:

- kernel_size – the size of the window

- stride – the stride of the window. Default value is kernel_size

- padding – implicit zero padding to be added on both sides

- ceil_mode – when True, will use ceil instead of floor to compute the output shape

- count_include_pad – when True, will include the zero-padding in the averaging calculation

#### <u>Activation</u>

Our activation module has ```ReSPro``` activation function implemented natively. By using <u>Re</u>flections, <u>S</u>calings and <u>Pro</u>jections on an hypersphere in higher dimensions, we created a non-linear differentiable associative activation function that can be represented in tensor form. It has only one parameter, that controls how close to linear or non-linear is the curve. More details are available on the main paper.

- alpha (```float```, optional) - controls the non-linearity of the curve. If it is not provided, it's automatically estimated.

#### <u>Regularization</u>

On regularization module we have ```Dropout``` implemented in this version. It is based on the idea of randomly shutting down some neurons in order to prevent overfitting. It takes only two parameters, listed below. This list was originally available in [PyTorch documentation page](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html?highlight=dropout#torch.nn.Dropout).  

- p – probability of an element to be zeroed. Default: 0.5

- inplace – If set to True, will do this operation in-place. Default: False


## 6. License

This software is licensed under the GNU General Public License v3.0. See the [`LICENSE`](LICENSE) file for details.
