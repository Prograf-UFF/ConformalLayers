from typing import List, Optional
import argparse, inspect, multiprocessing, os
import stuff
import torch


DEFAULT_WANDB_PROJECT = 'ConformalLayers - Benchmark'
DEFAULT_NUM_TRIALS = 50

DEFAULT_GPUS = -1 if torch.cuda.is_available() else 0  # -1 stands for all GPUs available, and 0 stands for CPU (no GPU).
DEFAULT_NUM_WORKERS = min(4, multiprocessing.cpu_count())  # 0 loads the data in the main process.

DEFAULT_SEED = 1992


def _range_check(values: Optional[List[int]]) -> Optional[List[int]]:
    if values is not None:
        if len(values) == 1:
            values = [values[0], values[0] + 1, 1]
        if not (1 <= len(values) <= 3):
            raise argparse.ArgumentTypeError('one, two or three int values expected')
    return values


def main(args: argparse.Namespace) -> None:
    model_class = eval(f'stuff.models.{args.model}')
    # Setup.
    config = stuff.make_basic_benchmark_config(name=model_class.__name__, program=os.path.basename(__file__))
    # Start a new sweep.
    stuff.benchmark(
        # Weights & Biases arguments.
        config=config,
        entity_name=args.wandb_entity,
        project_name=args.wandb_project,
        # Network and dataset arguments.
        model_class=model_class,
        batch_size_range=args.batch_size,
        depth_range=args.depth,
        # Run arguments.
        gpus=args.gpus,
        num_trials=args.num_trials,
        num_workers=args.num_workers,
        # Other arguments.
        seed=args.seed,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Network and dataset arguments.
    MODEL_CHOICES = [name for name, obj in inspect.getmembers(stuff.models) if inspect.isclass(obj) and issubclass(obj, stuff.models.ClassificationModel) and not inspect.isabstract(obj)]
    group = parser.add_argument_group('network arguments')
    group.add_argument('--model', metavar='CLASS_NAME', type=str, choices=MODEL_CHOICES, required=True, help='the name of the class of the model used in the benchmark')
    group.add_argument('--batch_size', metavar='VALUE | START STOP [STEP]', type=int, nargs='*', required=True, help='the value or range of values assumed by the size of the batch')
    group.add_argument('--depth', metavar='VALUE | START STOP [STEP]', type=int, nargs='*', help='the value or range of values assumed by the network''s depth (used only with models having varying depth)')
    # Weights & Biases arguments.
    group = parser.add_argument_group('logger arguments')
    group.add_argument('--wandb_entity', metavar='NAME', type=str, required=True, help='the name of the entity in the Weights & Biases framework')
    group.add_argument('--wandb_project', metavar='NAME', type=str, default=DEFAULT_WANDB_PROJECT, help='the name of the project in the Weights & Biases framework')
    # Run arguments.
    group = parser.add_argument_group('run arguments')
    group.add_argument('--gpus', metavar='COUNT', type=int, default=DEFAULT_GPUS, help=f'the number of GPUs used to train (0-{torch.cuda.device_count()}), or -1 to all')
    group.add_argument('--num_trials', metavar='COUNT', type=int, default=DEFAULT_NUM_TRIALS, help='the number of trials to run')
    group.add_argument('--num_workers', metavar='COUNT', type=int, default=DEFAULT_NUM_WORKERS, help=f'the number of workers used by the DataLoader (1-{multiprocessing.cpu_count()}), or 0 to load the data in the main process')
    # Other arguments.
    group = parser.add_argument_group('other arguments')
    group.add_argument('--seed', metavar='VALUE', type=int, default=DEFAULT_SEED, help='the seed for generating random numbers while splitting the dataset and performing data augmentation')
    # Parse arguments.
    args = parser.parse_args()
    args.batch_size = _range_check(args.batch_size)
    args.depth = _range_check(args.depth)
    # Call the main method.
    main(args)
