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
from adabelief_pytorch import AdaBelief

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import cl

from Experiments.utils.reporter import Reporter
from Experiments.utils.stopwatch import Stopwatch
from Experiments.archs2D import CLLenet
from Experiments.datasets import MNIST


# Device parameters
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
if DEVICE.type == 'cuda':
    torch.cuda.set_device(DEVICE)
else:
    print('Warning: The device was set to CPU.')


# Experiment parameters
EXPERIMENT_NAME = "CL_LeNet_2D"
EPOCHS = 300
METRICS_PATH = os.path.join('..', 'Results')
MODELS_PATH = os.path.join('..', 'Models')
if not os.path.exists(METRICS_PATH):
    os.makedirs(METRICS_PATH)
if not os.path.exists(MODELS_PATH):
    os.makedirs(MODELS_PATH)
net = CLLenet().to(DEVICE)
reporter = Reporter(os.path.join(METRICS_PATH, EXPERIMENT_NAME + '.csv'))


# Sets the seed for reproducibility
torch.manual_seed(1992)
np.random.seed(1992)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Dataset parameters
BATCHSIZE = 4096
shuffle = True
normalize = False
trainloader, testloader = MNIST(normalize, shuffle).get_dataset(BATCHSIZE)


# Train parameters
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(net.parameters())


# Train Loop
def train(epoch, optimizer):
    print('\nEpoch: %d' % epoch)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        optimizer.zero_grad()
        if DEVICE.type == 'cuda':
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        torch.cuda.reset_peak_memory_stats()
        start.record()
        with Stopwatch('Train -- Epoch {epoch}, Batch {batch_idx} -- Forward        -- Elapsed time: {et_str}.', {'epoch': epoch, 'batch_idx': batch_idx}):
            outputs = net(inputs)

        with Stopwatch('Train -- Epoch {epoch}, Batch {batch_idx} -- Loss           -- Elapsed time: {et_str}.', {'epoch': epoch, 'batch_idx': batch_idx}):
            loss = criterion(outputs, targets)

        with Stopwatch('Train -- Epoch {epoch}, Batch {batch_idx} -- Backward       -- Elapsed time: {et_str}.', {'epoch': epoch, 'batch_idx': batch_idx}):
            loss.backward()
            ##[ConformalLayers Promise] One has to set retain_graph=True while calling loss.backward() to keep the graph used to compute data cached by ConformalLayer objects

        with Stopwatch('Train -- Epoch {epoch}, Batch {batch_idx} -- Optimizer Step -- Elapsed time: {et_str}.', {'epoch': epoch, 'batch_idx': batch_idx}):
            optimizer.step()
        ##[ConformalLayers Promise] net.features.invalidate_cache() is not needed if optimizer.step() is able to update the data cached by ConformalLayer objects
        net.features.invalidate_cache()

        train_loss += loss.detach().item()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
       
        end.record()
        torch.cuda.synchronize()
        time = start.elapsed_time(end)
        memory = torch.cuda.max_memory_allocated()
        reporter.report(operation="Train", epoch=epoch, batch=batch_idx, loss=train_loss/(batch_idx+1), accuracy=correct/total, time=time, memory=memory)



# Test Loop
@torch.no_grad()
def test(epoch):    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    net.eval()
    test_loss, correct, total = 0, 0, 0
    loss_arr, acc_arr = [], []
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        torch.cuda.reset_peak_memory_stats()
        start.record()
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

        end.record()
        torch.cuda.synchronize()
        time = start.elapsed_time(end)
        memory = torch.cuda.max_memory_allocated()

        reporter.report(operation="Test", epoch=epoch, batch=batch_idx, loss=test_loss/(batch_idx+1), accuracy=correct/total, time=time, memory=memory)


# Main loop
for epoch in range(0, EPOCHS):
    train(epoch, optimizer)
    test(epoch)
    torch.save(net.state_dict(), os.path.join(MODELS_PATH, EXPERIMENT_NAME + ".pth"))
