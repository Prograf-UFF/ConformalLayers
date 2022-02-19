import argparse, inspect, multiprocessing, os
import stuff
import torch


DEFAULT_DATASETS_ROOT = os.path.relpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datasets'), '.')
DEFAULT_TRAIN_SIZE = 0.8

DEFAULT_WANDB_PROJECT = 'ConformalLayers - Sweep'
DEFAULT_WANDB_NUM_TRIALS = 200

DEFAULT_GPUS = -1 if torch.cuda.is_available() else 0  # -1 stands for all GPUs available, and 0 stands for CPU (no GPU).
DEFAULT_NUM_WORKERS = min(4, multiprocessing.cpu_count())  # 0 loads the data in the main process.

DEFAULT_BATCH_SIZE_VALUES = [32, 64, 128, 256, 512, 1024]
DEFAULT_EARLY_STOPPING_PATIENCE = 20
DEFAULT_LEARNING_RATE_RANGE = [0.0001, 0.001]
DEFAULT_LOG_FREQUENCY = 10
DEFAULT_OPTIMIZER_VALUES = ['Adam', 'RMSprop', 'SGD']
DEFAULT_SCHEDULER_PATIENCE = 10

DEFAULT_SEED = 1992


def main(args: argparse.Namespace) -> None:
    model_class = eval(f'stuff.models.{args.model}')
    datamodule_class = eval(f'stuff.datamodules.{args.datamodule}')
    if args.wandb_start:
        # Setup.
        config = stuff.make_basic_sweep_config(name=args.wandb_start, program=os.path.basename(__file__))
        config['metric'].update(model_class.wandb_metric())
        config['parameters'].update(stuff.make_parameter_configuration('batch_size', set(DEFAULT_BATCH_SIZE_VALUES)))
        config['parameters'].update(stuff.make_parameter_configuration('learning_rate', tuple(args.learning_rate_range)))
        config['parameters'].update(stuff.make_parameter_configuration('optimizer', set(DEFAULT_OPTIMIZER_VALUES)))
        # Start a new sweep.
        stuff.start_sweep(
            # Weights & Biases arguments.
            config=config,
            entity_name=args.wandb_entity,
            project_name=args.wandb_project,
            num_trials=args.wandb_num_trials,
            # Network and dataset arguments.
            model_class=model_class,
            datamodule_class=datamodule_class,
            datasets_root=args.datasets_root,
            train_size=args.train_size,
            # Run arguments.
            gpus=args.gpus,
            num_workers=args.num_workers,
            # Training arguments.
            early_stopping_patience=args.early_stopping_patience,
            log_frequency=args.log_frequency,
            scheduler_patience=args.scheduler_patience,
            # Other arguments.
            seed=args.seed,
        )
    else:
        # Resume an existing sweep.
        stuff.resume_sweep(
            # Weights & Biases arguments.
            sweep_id=args.wandb_resume,
            entity_name=args.wandb_entity,
            project_name=args.wandb_project,
            num_trials=args.wandb_num_trials,
            # Network and dataset arguments.
            model_class=model_class,
            datamodule_class=datamodule_class,
            datasets_root=args.datasets_root,
            train_size=args.train_size,
            # Run arguments.
            gpus=args.gpus,
            num_workers=args.num_workers,
            # Training arguments.
            early_stopping_patience=args.early_stopping_patience,
            log_frequency=args.log_frequency,
            scheduler_patience=args.scheduler_patience,
            # Other arguments.
            seed=args.seed,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Network and dataset arguments.
    MODEL_CHOICES = [name for name, obj in inspect.getmembers(stuff.models) if inspect.isclass(obj) and issubclass(obj, stuff.models.ClassificationModel) and not inspect.isabstract(obj)]
    DATAMODULE_CHOICES = [name for name, obj in inspect.getmembers(stuff.datamodules) if inspect.isclass(obj) and issubclass(obj, stuff.datamodules.ClassificationDataModule) and not inspect.isabstract(obj)]
    group = parser.add_argument_group('model and data module arguments')
    group.add_argument('--model', metavar='CLASS_NAME', type=str, choices=MODEL_CHOICES, required=True, help='the name of the class of the model used to train a network using hyperparameter sweep')
    group.add_argument('--datamodule', metavar='CLASS_NAME', type=str, choices=DATAMODULE_CHOICES, required=True, help='the name of the class of the data module used to load datasets')
    group.add_argument('--datasets_root', metavar='PATH', type=str, default=DEFAULT_DATASETS_ROOT, help='the path to the root folder of the datasets')
    group.add_argument('--train_size', metavar='PROPORTION', type=float, default=DEFAULT_TRAIN_SIZE, help='should be between 0.0 and 1.0 and represent the proportion of the fit dataset to include in the train split')
    # Weights & Biases arguments.
    group = parser.add_argument_group('logger arguments')
    switch = group.add_mutually_exclusive_group()
    switch.add_argument('--wandb_start', metavar='NAME', type=str, help='the name of the sweep to be created in the Weights & Biases framework')
    switch.add_argument('--wandb_resume', metavar='ID', type=str, help='the ID of the sweep to be resumed in the Weights & Biases framework')
    group.add_argument('--wandb_entity', metavar='NAME', type=str, required=True, help='the name of the entity in the Weights & Biases framework')
    group.add_argument('--wandb_project', metavar='NAME', type=str, default=DEFAULT_WANDB_PROJECT, help='the name of the project in the Weights & Biases framework')
    group.add_argument('--wandb_num_trials', metavar='COUNT', type=int, default=DEFAULT_WANDB_NUM_TRIALS, help='the number of trials to run')
    # Run arguments.
    group = parser.add_argument_group('run arguments')
    group.add_argument('--gpus', metavar='COUNT', type=int, default=DEFAULT_GPUS, help=f'the number of GPUs used to train (0-{torch.cuda.device_count()}), or -1 to all')
    group.add_argument('--num_workers', metavar='COUNT', type=int, default=DEFAULT_NUM_WORKERS, help=f'the number of workers used by the DataLoader (1-{multiprocessing.cpu_count()}), or 0 to load the data in the main process')
    # Training arguments.
    group = parser.add_argument_group('training arguments')
    group.add_argument('--early_stopping_patience', metavar='EPOCHS', type=int, default=DEFAULT_EARLY_STOPPING_PATIENCE, help='number of epochs with no improvement after which the training is stoped')
    group.add_argument('--learning_rate_range', metavar='MIN MAX', type=float, nargs=2, default=DEFAULT_LEARNING_RATE_RANGE, help='the range of values assumed by the `learning_rate` hyperparameter')
    group.add_argument('--log_frequency', metavar='STEPS', type=int, default=DEFAULT_LOG_FREQUENCY, help='how often to log within steps')
    group.add_argument('--scheduler_patience', metavar='EPOCHS', type=int, default=DEFAULT_SCHEDULER_PATIENCE, help='number of epochs with no improvement after which learning rate will be reduced')
    # Other arguments.
    group = parser.add_argument_group('other arguments')
    group.add_argument('--seed', metavar='VALUE', type=int, default=DEFAULT_SEED, help='the seed for generating random numbers')
    # Parse arguments.
    args = parser.parse_args()
    if args.wandb_resume is None and args.wandb_start is None:
        args.wandb_start = f'{args.model} on {args.datamodule}'
    # Call the main method.
    main(args)
