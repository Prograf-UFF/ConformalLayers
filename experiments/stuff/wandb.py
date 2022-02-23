from .datamodules import ClassificationDataModule, RandomDataModule
from .models import ClassificationModel
from pandas import DataFrame
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union
import csv, gc, os, tempfile
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import wandb


BATCH_SIZE_FIELD = 'batch_size'
DEPTH_FIELD = 'depth'
ELAPSED_TIME_FIELD = 'elapsed_time'
EMISSION_FIELD = 'emission'
INDEX_FIELD = 'index'
MAX_MEMORY_FIELD = 'max_memory'
MEAN_ELAPSED_TIME_FIELD = 'mean_elapsed_time'
MEAN_EMISSION_FIELD = 'mean_emission'
MEAN_MAX_MEMORY_FIELD = 'mean_max_memory'
NAME_FIELD = 'name'
STD_ELAPSED_TIME_FIELD = 'std_elapsed_time'
STD_EMISSION_FIELD = 'std_mean_emission'
STD_MAX_MEMORY_FIELD = 'std_max_memory'

BENCHMARK_CSV_FILENAME = 'benchmark.csv'
BENCHMARK_CSV_FIELDS = [INDEX_FIELD, DEPTH_FIELD, BATCH_SIZE_FIELD, ELAPSED_TIME_FIELD, MAX_MEMORY_FIELD, EMISSION_FIELD]

BENCHMARK_SUMMARY_FIELDS = [NAME_FIELD, DEPTH_FIELD, BATCH_SIZE_FIELD, MEAN_ELAPSED_TIME_FIELD, STD_ELAPSED_TIME_FIELD, MEAN_MAX_MEMORY_FIELD, STD_MAX_MEMORY_FIELD, MEAN_EMISSION_FIELD, STD_EMISSION_FIELD]


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
        batch_size_range: List[int],
        depth_range: Optional[List[int]],
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
        # Ensure full reproducibility.
        if seed is not None:
            pl.seed_everything(seed, workers=True)
        # Setup the trainer.
        trainer = pl.Trainer(deterministic=seed is not None, gpus=gpus, log_every_n_steps=1, logger=pl.loggers.WandbLogger(experiment=run), num_sanity_val_steps=0)
        # Run experiments.
        with open(os.path.join(run.dir, BENCHMARK_CSV_FILENAME), mode='w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=BENCHMARK_CSV_FIELDS)
            writer.writeheader()
            for batch_size in range(*batch_size_range):
                depths = [None] if depth_range is None else range(*depth_range)
                for depth in depths:
                    datamodule = RandomDataModule(batch_size=batch_size, **wandb_args, **kwargs)
                    model = model_class(depth=depth, run_dir=run.dir, **wandb_args, **kwargs)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    writer.writerows(trainer.predict(model, datamodule=datamodule))


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


def summarize_benchmarks(entity_name: str, project_name: str) -> DataFrame:
    # Start W&B API and get all runs from the target project.
    api = wandb.Api()
    runs = api.runs(f'{entity_name}/{project_name}')
    # Sumarize benchmark data downloades from CSV files.
    mean_and_std = lambda column: (column.mean().item(), column.std().item())
    summary = []
    for run in runs:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run.file(BENCHMARK_CSV_FILENAME).download(root=tmp_dir, replace=True)
            df = pd.read_csv(os.path.join(tmp_dir, BENCHMARK_CSV_FILENAME))
        keys = df[[DEPTH_FIELD, BATCH_SIZE_FIELD]].drop_duplicates()
        for _, (depth, batch_size) in keys.iterrows():
            subset = df[(df[DEPTH_FIELD] == depth) & (df[BATCH_SIZE_FIELD] == batch_size) & (df[INDEX_FIELD] != 0)]
            mean_elapsed_time, std_elapsed_time = mean_and_std(subset[ELAPSED_TIME_FIELD])
            mean_max_memory, std_max_memory = mean_and_std(subset[MAX_MEMORY_FIELD])
            mean_emission, std_emission = mean_and_std(subset[EMISSION_FIELD])
            summary.append({
                NAME_FIELD: run.config['name'],
                DEPTH_FIELD: depth,
                BATCH_SIZE_FIELD: batch_size,
                MEAN_ELAPSED_TIME_FIELD: mean_elapsed_time,
                STD_ELAPSED_TIME_FIELD: std_elapsed_time,
                MEAN_MAX_MEMORY_FIELD: mean_max_memory,
                STD_MAX_MEMORY_FIELD: std_max_memory,
                MEAN_EMISSION_FIELD: mean_emission,
                STD_EMISSION_FIELD: std_emission,
            })
    # Return summarized data.
    summary = pd.DataFrame(summary, columns=BENCHMARK_SUMMARY_FIELDS)
    summary.sort_values([NAME_FIELD, DEPTH_FIELD, BATCH_SIZE_FIELD], inplace=True)
    return summary
