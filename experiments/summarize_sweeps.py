from pandas import DataFrame
from sweep import DEFAULT_WANDB_PROJECT
from typing import Optional
import argparse, os, re
import pandas as pd
import stuff


DEFAULT_OUTPUT = 'results'

MEAN_CORRUPTION_ERROR_FIELD = 'mean_corruption_error'
MEAN_RELATIVE_CORRUPTION_ERROR_FIELD = 'relative_mean_corruption_error'


def append_cifar10c_corruption_metrics(df: DataFrame, baseline_network: Optional[str] = None) -> DataFrame:
    TEST_ACCURACY_FIELD_PATTERN = re.compile(stuff.wandb.TEST_ACCURACY_FIELD_MASK.format('.*'))
    TEST_ACCURACY_CLEAN_FIELD = stuff.wandb.TEST_ACCURACY_FIELD_MASK.format('clean')
    TEST_ACCURACY_CORRUPTION_SEVERITY_FIELDS = list(filter(lambda arg: arg != TEST_ACCURACY_CLEAN_FIELD and bool(TEST_ACCURACY_FIELD_PATTERN.match(arg)), df.columns))
    TEST_ACCURACY_CORRUPTION_FIELDS = set(map(lambda arg: arg[:-2], TEST_ACCURACY_CORRUPTION_SEVERITY_FIELDS))  # By construction, fields in TEST_ACCURACY_CORRUPTION_SEVERITY_FIELDS end with '_{level}', where 1 <= level <= 5.
    # Select only CIFAR-10 entries.
    cifar_df = df[df[stuff.wandb.DATASET_FIELD] == 'CIFAR10']
    if baseline_network is None:
        baseline_network = cifar_df[stuff.wandb.NETWORK_FIELD][cifar_df[TEST_ACCURACY_CLEAN_FIELD].idxmin()]
    baseline_row = cifar_df[stuff.wandb.NETWORK_FIELD] == baseline_network
    # Compute metrics.
    clean_error_rate = 1 - cifar_df[TEST_ACCURACY_CLEAN_FIELD]
    corruption_error = pd.DataFrame(columns=TEST_ACCURACY_CORRUPTION_FIELDS)
    relative_corruption_error = pd.DataFrame(columns=TEST_ACCURACY_CORRUPTION_FIELDS)
    for curr_corruption_field in TEST_ACCURACY_CORRUPTION_FIELDS:
        curr_corruption_severity_fields = list(filter(lambda arg: arg.startswith(curr_corruption_field), TEST_ACCURACY_CORRUPTION_SEVERITY_FIELDS))
        curr_corruption_error_rates = 1 - cifar_df[curr_corruption_severity_fields]
        curr_corruption_error_rates_mean = curr_corruption_error_rates.mean(axis='columns')
        corruption_error[curr_corruption_field] = curr_corruption_error_rates_mean / curr_corruption_error_rates_mean[baseline_row].item()
        relative_corruption_error[curr_corruption_field] = (curr_corruption_error_rates_mean - clean_error_rate) / (curr_corruption_error_rates_mean[baseline_row] - clean_error_rate[baseline_row]).item()
    mean_corruption_error = corruption_error.mean(axis='columns')
    mean_relative_corruption_error = relative_corruption_error.mean(axis='columns')
    # Return computed values.
    return pd.concat((df, mean_corruption_error.rename(MEAN_CORRUPTION_ERROR_FIELD), mean_relative_corruption_error.rename(MEAN_RELATIVE_CORRUPTION_ERROR_FIELD)), axis='columns')

def snake_to_camel(snake: str):
    return ''.join([word.title() for word in snake.split('_')])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Weights & Biases arguments.
    group = parser.add_argument_group('logger arguments')
    group.add_argument('--wandb_entity', metavar='NAME', type=str, required=True, help='the name of the entity in the Weights & Biases framework')
    group.add_argument('--wandb_project', metavar='NAME', type=str, default=DEFAULT_WANDB_PROJECT, help='the name of the project in the Weights & Biases framework')
    # Result arguments.
    group.add_argument('--output_dir', metavar='FILENAME', type=str, default=DEFAULT_OUTPUT, help='the path to the folder where resulting CSV will be written')
    # Parse arguments.
    args = parser.parse_args()
    # Call the magic procedure to download results from W&B.
    df = stuff.summarize_sweeps(entity_name=args.wandb_entity, project_name=args.wandb_project)
    df = append_cifar10c_corruption_metrics(df)
    # Rename columns to LaTeX-friendly names.
    df.rename(columns=dict(map(lambda snake: (snake, snake_to_camel(snake)), df.columns)), inplace=True)
    # Save the complete set of results.
    os.makedirs(args.output_dir, exist_ok=True)
    df.to_csv(os.path.join(args.output_dir, 'sweep.csv'), index=False)
    # Save results by dataset.
    DATASET_FIELD = snake_to_camel(stuff.wandb.DATASET_FIELD)
    datasets = df[[DATASET_FIELD]].drop_duplicates()
    for _, (dataset,) in datasets.iterrows():
        df[(df[DATASET_FIELD] == dataset)].to_csv(os.path.join(args.output_dir, f'sweep_{dataset}.csv'), index=False)
