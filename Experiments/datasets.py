import torch
import torchvision
import torchvision.transforms as transforms  
import os


class MNIST:
    def __init__(self, normalize=False, shuffle=False, add_channel=True, pad=True) -> None:
        super().__init__()
        self.normalize = normalize
        self.shuffle = shuffle
        self.add_channel = add_channel
        self.pad = pad

    def get_dataset(self, batchsize):
        if self.pad:
            t = [transforms.Pad(2), transforms.ToTensor()]
        else:
            t = [transforms.ToTensor()]

        if self.normalize:
            t.append(transforms.Normalize((0.1307,), (0.3081,)))

        if self.add_channel:
            t.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
        transform = transforms.Compose(t)
        
        dataset1 = torchvision.datasets.MNIST(os.path.join('..', 'Datasets'), train=True, download=True,
                                              transform=transform)
        dataset2 = torchvision.datasets.MNIST(os.path.join('..', 'Datasets'), train=False, transform=transform)
        
        trainloader = torch.utils.data.DataLoader(dataset1, batch_size=batchsize, shuffle=self.shuffle)
        testloader = torch.utils.data.DataLoader(dataset2, batch_size=batchsize, shuffle=self.shuffle)

        return trainloader, testloader


class FashionMNIST:
    def __init__(self, normalize=False, shuffle=False, oversample=None) -> None:
        super().__init__()
        self.normalize = normalize
        self.shuffle = shuffle
        self.oversample = oversample

    def get_dataset(self, batchsize):
        if self.oversample:
            pad = 2 + self.oversample
        else:
            pad = 2
        t = [transforms.Pad(pad), transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(3, 1, 1))]
        if self.normalize:
            t.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        
        transform = transforms.Compose(t)
        
        dataset1 = torchvision.datasets.FashionMNIST(os.path.join('..', 'Datasets'), train=True, download=True,
                                                     transform=transform)
        dataset2 = torchvision.datasets.FashionMNIST(os.path.join('..', 'Datasets'), train=False, transform=transform)
        
        trainloader = torch.utils.data.DataLoader(dataset1, batch_size=batchsize, shuffle=self.shuffle)
        testloader = torch.utils.data.DataLoader(dataset2, batch_size=batchsize, shuffle=self.shuffle)

        return trainloader, testloader


class CIFAR10:
    def __init__(self, normalize=False, shuffle=False) -> None:
        super().__init__()
        self.normalize = normalize
        self.shuffle = shuffle

    def get_dataset(self, batchsize):
        t = [transforms.ToTensor()]
        if self.normalize:
            t.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)))
        
        transform = transforms.Compose(t)
        
        dataset1 = torchvision.datasets.CIFAR10(os.path.join('..', 'Datasets'), train=True, download=True,
                                                transform=transform)
        dataset2 = torchvision.datasets.CIFAR10(os.path.join('..', 'Datasets'), train=False, transform=transform)
        
        trainloader = torch.utils.data.DataLoader(dataset1, batch_size=batchsize, shuffle=self.shuffle)
        testloader = torch.utils.data.DataLoader(dataset2, batch_size=batchsize, shuffle=self.shuffle)

        return trainloader, testloader
        