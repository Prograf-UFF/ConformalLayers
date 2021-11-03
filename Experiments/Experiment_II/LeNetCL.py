import argparse, os, sys, warnings
import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from Experiments.utils import progress_bar
from Experiments.networks.lenet import LeNetCL


def train(net, trainloader, criterion, device, optimizer):
    net.train()
    train_loss, correct, total = 0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        optimizer.zero_grad()
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.detach().item()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)\n' % (
            train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


@torch.no_grad()
def test(net, testloader, criterion, device):
    net.eval()
    test_loss, correct, total = 0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.detach().item()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)\n' % (
            test_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def main():
    # Device parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        warnings.warn(f'The device was set to {device}.', RuntimeWarning)

    # Set the seeds for reproducibility
    torch.manual_seed(1992)
    np.random.seed(1992)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', help="num of epochs", default=300, type=int)
    parser.add_argument('--batch_size', help="batch size", default=4096, type=int)
    parser.add_argument('--learning_rate', help="learning_rate", default=0.001, type=float)
    parser.add_argument('--optimizer', help="optimizer", default='adam', type=str)
    parser.add_argument('--dataset', help="dataset", default='MNIST', type=str)
    parser.add_argument('--dropout', help="dropout", default=0.5, type=float)
    args = parser.parse_args()

    net = LeNetCL().to(device)
    criterion = nn.CrossEntropyLoss()

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(net.parameters(), lr=args.learning_rate)

    if args.dataset == 'MNIST':
        from Experiments.datasets import MNIST
        trainloader, testloader = MNIST(normalize=True, shuffle=True).get_dataset(args.batch_size)
    elif args.dataset == 'FashionMNIST':
        from Experiments.datasets import FashionMNIST
        trainloader, testloader = FashionMNIST(normalize=True, shuffle=True).get_dataset(args.batch_size)
    elif args.dataset == 'CIFAR10':
        from Experiments.datasets import CIFAR10
        trainloader, testloader = CIFAR10(normalize=True, shuffle=True).get_dataset(args.batch_size)

    # Train/Inference main loop
    for _ in range(0, args.epochs + 1):
        train(net, trainloader, criterion, device, optimizer)
        test(net, testloader, criterion, device)


if __name__ == "__main__":
    main()
