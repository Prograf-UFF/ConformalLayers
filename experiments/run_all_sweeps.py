from stuff.datamodules import CIFAR10, FashionMNIST, MNIST
from stuff.models import BaseLinearNet, BaseReLUNet, BaseReSProNet, LeNet, LeNetCL
from tqdm import tqdm
import argparse, os, subprocess, sys


DATAMODULES = [CIFAR10, FashionMNIST, MNIST]
MODELS = [BaseLinearNet, BaseReLUNet, BaseReSProNet, LeNet, LeNetCL]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Weights & Biases arguments.
    group = parser.add_argument_group('logger arguments')
    group.add_argument('--wandb_entity', metavar='NAME', type=str, required=True, help='the name of the entity in the Weights & Biases framework')
    # Parse arguments.
    args = parser.parse_args()
    # Run experiment.
    pbar = tqdm(desc='Sweeps', total=(len(DATAMODULES) * len(MODELS)))
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sweep.py')
    for datamodule_class in DATAMODULES:
        for model_class in MODELS:
            subprocess.run([sys.executable, script_path, '--model', model_class.__name__, '--datamodule', datamodule_class.__name__, '--wandb_entity', args.wandb_entity])
            pbar.update(1)
