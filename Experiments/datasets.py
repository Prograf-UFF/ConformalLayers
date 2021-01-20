import torch
import torchvision
import torchvision.transforms as transforms  

class MNIST():
    def __init__(self, normalize=False, shuffle=False) -> None:
        super().__init__()
        self.normalize = normalize
        self.shuffle = shuffle

    def get_dataset(self, batchsize):
        t = [transforms.ToTensor()]
        if self.normalize:
            t.append(transforms.Normalize((0.1307,), (0.3081,)))
        
        transform = transforms.Compose(t)
        
        dataset1 = torchvision.datasets.MNIST('./Datasets', train=True, download=True, transform=transform)
        dataset2 = torchvision.datasets.MNIST('./Datasets', train=False, transform=transform)
        
        trainloader = torch.utils.data.DataLoader(dataset1, batch_size=batchsize, shuffle=self.shuffle)
        testloader = torch.utils.data.DataLoader(dataset2, batch_size=batchsize, shuffle=self.shuffle)

        return trainloader, testloader
