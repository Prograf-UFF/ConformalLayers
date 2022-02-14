import argparse
import glob
import numpy as np
import os
import pprint
import torch
import torchvision
import tqdm

from glob import glob
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils import load_txt, accuracy, create_barplot, get_fname, AverageMeter
from Experiments.networks.baseline import *
from Experiments.networks.lenet import *
from dataset import CIFAR10C
import csv


# CIFAR10
CORRUPTIONS = load_txt('./corruptions.txt')
MEAN=[0.485, 0.456, 0.406]
STD=[0.229, 0.224, 0.225]


def main(opt):

    device = torch.device('cuda:0')

    # model
    if opt.arch == 'baselinearnet':
        model = BaseLinearNet()
        weight_path = os.path.join(os.path.dirname(__file__), '..', 'Experiment_I', 'baselinearnet.pth')
    elif opt.arch == 'baserespronet':
        model = BaseReSProNet()
        weight_path = os.path.join(os.path.dirname(__file__), '..', 'Experiment_I', 'baserespronet.pth')
    elif opt.arch == 'baserelunet':
        model = BaseReLUNet()
        weight_path = os.path.join(os.path.dirname(__file__), '..', 'Experiment_I', 'baserelunet.pth')
    elif opt.arch == 'lenetcl':
        model = LeNetCL()
        weight_path = os.path.join(os.path.dirname(__file__), 'LeNetCL.pth')
    elif opt.arch == 'lenet':
        model = LeNet()
        weight_path = os.path.join(os.path.dirname(__file__), 'LeNet.pth')
    else:
        raise ValueError()
    try:
        model.load_state_dict(torch.load(weight_path, map_location='cpu'))
    except:
        model.load_state_dict(torch.load(weight_path, map_location='cpu')['model'])
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    # Main inference loop
    error_rates = []

    filename = opt.arch + '.csv'
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['DistortionName', 'Severity', 'Error'])

        for _, cname in enumerate(CORRUPTIONS):
            local_errs = []
            for severity in range(1, 6):
                # load dataset
                if cname == 'natural':
                    dataset = datasets.CIFAR10(
                        os.path.join(opt.data_root, 'cifar10'),
                        train=False, transform=transform, download=True,
                    )
                else:
                    dataset = CIFAR10C(
                        os.path.join(opt.data_root, 'cifar10-c'), 
                        cname, severity, transform=transform
                    )
                loader = DataLoader(dataset, batch_size=opt.batch_size,
                                    shuffle=False)

                correct = 0
                with torch.no_grad():
                    for itr, (x, y) in enumerate(loader):
                        x = x.to(device, non_blocking=True)
                        y = y.to(device, dtype=torch.int64, non_blocking=True)

                        yhat = model(x)

                        pred = yhat.data.max(1)[1].cpu()
                        correct += pred.eq(y.cpu()).sum()

                local_errs.append(1 - 1. * correct / len(dataset))
                writer.writerow([cname, severity, local_errs[-1].item()])
            print('\n=Average', tuple(local_errs))
            print(np.mean(local_errs))

            rate = np.mean(local_errs)
            error_rates.append(rate)
            print('Distortion: {:15s}  | CE (unnormalized) (%): {:.2f}'.format(cname, 100 * rate))
    print('mCE (unnormalized by AlexNet errors) (%): {:.2f}'.format(100 * np.mean(error_rates)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--arch',
        type=str, default='baselinearnet',
        help='model name'
    )
    parser.add_argument(
        '--data_root',
        type=str, default='.',
        help='root path to cifar10-c directory'
    )
    parser.add_argument(
        '--batch_size',
        type=int, default=1024,
        help='batch size',
    )

    opt = parser.parse_args()

    main(opt)