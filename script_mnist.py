import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms  
import os
import argparse
import sys
import numpy as np

try:
    import cl
except ModuleNotFoundError:
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    import cl
from Experiments.utils.utils import progress_bar


# Defines a NN topology
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.features = cl.ConformalLayers(
            cl.Conv2d(in_channels=1, out_channels=64, kernel_size=5),
            cl.AvgPool2d(kernel_size=2, stride=2),
            cl.SRePro(),
          )
        self.fc1 = nn.Linear(9216, 10)

    def forward(self, x):
        out = self.features(x)
        out = nn.Flatten()(out)
        out = self.fc1(out)
        return out


# Device to run the workload
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
if DEVICE.type == 'cuda':
    torch.cuda.set_device(DEVICE)
else:
    print('Warning: The device was set to CPU.')


# The size of the batch
BATCHSIZE = 16


# Sets the seed for reproducibility
torch.manual_seed(1992)
np.random.seed(1992)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def get_dataset():
    transform=transforms.Compose([
        # transforms.Resize((14, 14)),
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    dataset1 = torchvision.datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = torchvision.datasets.MNIST('../data', train=False,
                       transform=transform)
    
    trainloader = torch.utils.data.DataLoader(dataset1, batch_size=BATCHSIZE, shuffle=False)
    testloader = torch.utils.data.DataLoader(dataset2, batch_size=BATCHSIZE, shuffle=False)

    return trainloader, testloader


net = Network().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

trainloader, testloader = get_dataset()


def train(epoch, optimizer):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    loss_arr = []
    acc_arr = []
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        optimizer.zero_grad()
        if DEVICE.type == 'cuda':
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        loss_arr.append(loss.detach().item())
        train_loss += loss_arr[-1]

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        acc_arr.append(correct / total)

        loss.backward()
        optimizer.step()
        net.features.invalidate_cache()
   
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return loss_arr, acc_arr


@torch.no_grad()
def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    loss_arr = []
    acc_arr = []
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if DEVICE.type == 'cuda':
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        loss_arr.append(loss.detach().item())
        test_loss += loss_arr[-1]
        
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        acc_arr.append(correct / total)

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return loss_arr, acc_arr


train_data = []
test_data = []

for epoch in range(0, 10):

    train_data += train(epoch, optimizer)
    # with open('train_results.p', 'wb') as f:
    #     pickle.dump(train_data, f)

    test_data += test(epoch)
    # with open('test_results.p', 'wb') as f:
    #     pickle.dump(test_data, f)
