from .datamodules import ClassificationDataModule, RandomDataModule
from .models import ClassificationModel
from tqdm import tqdm
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Type, Union
import numpy as np
import os
import pandas as pd
import pytorch_lightning as pl
import wandb


def _tracked_sweep_run(*,
    # Network and dataset arguments.
    model_class: Type[ClassificationModel],
    datamodule_class: Type[ClassificationDataModule],
    # Run arguments.
    gpus: int,
    # Training arguments.
    log_frequency: int,
    seed: Optional[int],
    # Other arguments.
    **kwargs: Any
) -> None:
    # Start a new tracked run at Weights & Biases.
    with wandb.init(settings=wandb.Settings(start_method='fork')) as run:
        wandb_args = run.config.as_dict()
        # Ensure full reproducibility.
        if seed is not None:
            pl.seed_everything(seed, workers=True)
        # Setup the trainer.
        trainer = pl.Trainer(
            callbacks=[pl.callbacks.model_checkpoint.ModelCheckpoint(os.path.join(run.dir, 'checkpoint'), save_weights_only=True)],
            deterministic=seed is not None,
            gpus=gpus,
            log_every_n_steps=log_frequency,
            logger=pl.loggers.WandbLogger(experiment=run),
            num_sanity_val_steps=0,
        )
        # Setup the model and the data module.
        model = model_class(run_dir=run.dir, test_dataset_name=datamodule_class.test_dataset_name(), **wandb_args, **kwargs)
        datamodule = datamodule_class(run_dir=run.dir, **wandb_args, **kwargs)
        # Perform model fitting.
        trainer.fit(model, datamodule=datamodule)
        # Perform model test.
        trainer.test(model, datamodule=datamodule)


def benchmark(
        # W&B arguments.
        config: Dict[str, Any],
        entity_name: str,
        project_name: str,
        # Network and dataset arguments.
        model_class: Type[ClassificationModel],
        batch_size: Optional[int],
        batch_size_range: Optional[Union[Tuple[int, int], Tuple[int, int, int]]],
        depth_range: Optional[Union[Tuple[int, int], Tuple[int, int, int]]],
        # Run arguments.
        gpus: int,
        # Training arguments.
        seed: Optional[int],
        # Other arguments.
        **kwargs: Any
    ) -> None:
    # Start a new tracked run at Weights & Biases.
    with wandb.init(config=config, entity=entity_name, project=project_name, settings=wandb.Settings(start_method='fork')) as run:
        wandb_args = run.config.as_dict()
        results = []
        # Ensure full reproducibility.
        if seed is not None:
            pl.seed_everything(seed, workers=True)
        # Setup the trainer.
        trainer = pl.Trainer(deterministic=seed is not None, gpus=gpus, log_every_n_steps=1, logger=pl.loggers.WandbLogger(experiment=run), num_sanity_val_steps=0)
        # Benchmark changing the batch size.
        if batch_size_range is not None:
            model = model_class(run_dir=run.dir, **wandb_args, **kwargs)
            for batch_size in tqdm(range(*batch_size_range), desc='Batch size'):
                datamodule = RandomDataModule(batch_size=batch_size, **wandb_args, **kwargs)
                results.extend(trainer.predict(model, datamodule=datamodule))
        # Benchmark changing network's depth.
        elif depth_range is not None and batch_size is not None:
            datamodule = RandomDataModule(batch_size=batch_size, **wandb_args, **kwargs)
            for depth in tqdm(range(*depth_range), desc='Depth'):
                model = model_class(depth=depth, run_dir=run.dir, **wandb_args, **kwargs)
                results.extend(trainer.predict(model, datamodule=datamodule))
        # Invalid option.
        else:
            raise ValueError('One of the following set of arguments must be provided: `batch_size_range` or `depth_range` and `batch_size`.')
        # Log results.
        run.log({'benchmark': wandb.Table(dataframe=pd.DataFrame(results))})


def make_basic_benchmark_config(name: str, program: str) -> Dict[str, Any]:
    return {'name': name, 'program': program, 'parameters': {}}


def make_basic_sweep_config(name: str, program: str) -> Dict[str, Any]:
    return {'name': name, 'program': program, 'method': 'bayes', 'early_terminate': {'type': 'hyperband', 'min_iter': 10}, 'metric': {}, 'parameters': {}}


def make_parameter_configuration(name: str, value: Union[Any, Tuple[Any, Any], Tuple[Any, Any, Any]]) -> Dict[str, Dict[str, Any]]:
    if isinstance(value, set):
        return {name: {'values': list(sorted(value))}}
    else:
        value = tuple(value) if isinstance(value, Iterable) else (value,)
        if len(value) == 1 or value[0] == value[1]:
            return {name: {'value': value[0]}}
        elif len(value) == 2:
            return {name: {'min': value[0], 'max': value[1]}}
        elif len(value) == 3:
            return {name: {'values': list(map(lambda arg: arg.item(), np.arange(value[0], value[1] + value[2], value[2])))}}
    raise ValueError('Invalid value')


def resume_sweep(sweep_id: str, entity_name: str, project_name: str, num_trials: Optional[int] = None, **kwargs) -> None:
    wandb.agent(sweep_id, function=lambda: _tracked_sweep_run(**kwargs), entity=entity_name, project=project_name, count=num_trials)


def start_sweep(config: Dict[str, Any], entity_name: str, project_name: str, num_trials: Optional[int] = None, **kwargs: Any) -> None:
    sweep_id = wandb.sweep(config, entity=entity_name, project=project_name)
    wandb.agent(sweep_id, function=lambda: _tracked_sweep_run(**kwargs), count=num_trials)
