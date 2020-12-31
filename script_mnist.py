import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms  
import warnings

try:
    import cl
except ModuleNotFoundError:
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    import cl

from Experiments.utils.utils import progress_bar
from utils import Stopwatch


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
    warnings.warn('The device was set to CPU.', RuntimeWarning)


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
    
    dataset1 = torchvision.datasets.MNIST('./Datasets', train=True, download=True, transform=transform)
    dataset2 = torchvision.datasets.MNIST('./Datasets', train=False, transform=transform)
    
    trainloader = torch.utils.data.DataLoader(dataset1, batch_size=BATCHSIZE, shuffle=False)
    testloader = torch.utils.data.DataLoader(dataset2, batch_size=BATCHSIZE, shuffle=False)

    return trainloader, testloader


net = Network().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = cl.SGD(net.parameters(), [net.features], lr=0.01, momentum=0.9)  #[ConformalLayers Hint] One has to use an optimizer adapted for the ConformalLayers

trainloader, testloader = get_dataset()


def train(epoch, optimizer):
    net.train()
    train_loss, correct, total = 0, 0, 0
    loss_arr, acc_arr = [], []
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        optimizer.zero_grad()
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
        with Stopwatch('Train -- Epoch {epoch}, Batch {batch_idx} -- Forward        -- Elapsed time: {et_str}.', {'epoch': epoch, 'batch_idx': batch_idx}):
            outputs = net(inputs)

        with Stopwatch('Train -- Epoch {epoch}, Batch {batch_idx} -- Loss           -- Elapsed time: {et_str}.', {'epoch': epoch, 'batch_idx': batch_idx}):
            loss = criterion(outputs, targets)

        with Stopwatch('Train -- Epoch {epoch}, Batch {batch_idx} -- Backward       -- Elapsed time: {et_str}.', {'epoch': epoch, 'batch_idx': batch_idx}):
            loss.backward(retain_graph=True) #[ConformalLayers Hint] One has to set retain_graph=True while calling loss.backward() to keep the graph used to compute data cached by ConformalLayer objects

        with Stopwatch('Train -- Epoch {epoch}, Batch {batch_idx} -- Optimizer Step -- Elapsed time: {et_str}.', {'epoch': epoch, 'batch_idx': batch_idx}):
            optimizer.step()
        net.features.invalidate_cache()  #TODO Remover, caso a atualização da cache se mostre correta

        loss_arr.append(loss.detach().item())
        train_loss += loss_arr[-1]

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        acc_arr.append(correct / total)

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)\n'
                    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return loss_arr, acc_arr


@torch.no_grad()
def test(epoch):
    net.eval()
    test_loss, correct, total = 0, 0, 0
    loss_arr, acc_arr = [], []
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        with Stopwatch('Test -- Epoch {epoch}, Batch {batch_idx} -- Forward -- Elapsed time: {et_str}.', {'epoch': epoch, 'batch_idx': batch_idx}):
            outputs = net(inputs)

        with Stopwatch('Test -- Epoch {epoch}, Batch {batch_idx} -- Loss    -- Elapsed time: {et_str}.', {'epoch': epoch, 'batch_idx': batch_idx}):
            loss = criterion(outputs, targets)

        loss_arr.append(loss.detach().item())
        test_loss += loss_arr[-1]
        
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        acc_arr.append(correct / total)

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)\n'
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
