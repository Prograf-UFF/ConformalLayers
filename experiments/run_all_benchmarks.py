from stuff.networks import D3ModNet, D3ModNetCL, DkNet, DkNetCL, LeNet, LeNetCL
from tqdm import tqdm
import argparse, os, subprocess, sys


BATCH_SIZE_MODELS = [D3ModNet, D3ModNetCL, LeNet, LeNetCL]
DEPTH_MODELS = [DkNet, DkNetCL]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Weights & Biases arguments.
    group = parser.add_argument_group('logger arguments')
    group.add_argument('--wandb_entity', metavar='NAME', type=str, required=True, help='the name of the entity in the Weights & Biases framework')
    # Parse arguments.
    args = parser.parse_args()
    # Run experiment.
    pbar = tqdm(desc='Benchmarks', total=(len(BATCH_SIZE_MODELS) + len(DEPTH_MODELS)))
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'benchmark.py')
    for model_class in BATCH_SIZE_MODELS:
        subprocess.run([sys.executable, script_path, '--model', model_class.__name__, '--batch_size', '1000', '6000', '1000', '--wandb_entity', args.wandb_entity])
        pbar.update(1)
    for model_class in DEPTH_MODELS:
        subprocess.run([sys.executable, script_path, '--model', model_class.__name__, '--batch_size', '64', '--depth', '1', '13', '--wandb_entity', args.wandb_entity])
        pbar.update(1)
