from pandas import DataFrame
from scipy import stats
from sweep import DEFAULT_WANDB_PROJECT
from typing import OrderedDict
import argparse, os, re
import pandas as pd
import stuff


DEFAULT_OUTPUT = 'results'

CLEAN_ERROR_RATE_FIELD = 'clean_error_rate'
CORRUPTION_FIELDS: str = None  # Late-initialization.
CORRUPTION_FIELD = 'corruption'
MEAN_CORRUPTION_ERROR_MASK = 'mean_corruption_error_{}'  # The format is mean_corruption_error_{baseline-network}.
MEAN_CORRUPTION_ERROR_RATE_FIELD = 'mean_corruption_error_rate'
MEAN_CORRUPTION_ERROR_RATE_FIELD_MASK = 'mean_{}_error_rate'  # The format is mean_{corruption}_error_rate.
P_VALUE_FIELD_MASK = 'p_value_{}'  # The format is p_value_{compared-network}
RELATIVE_MEAN_CORRUPTION_ERROR_MASK = 'relative_mean_corruption_error_{}'  # The format is relative_mean_corruption_error_{baseline-network}.
STD_CORRUPTION_ERROR_RATE_FIELD = 'std_corruption_error_rate'
STD_CORRUPTION_ERROR_RATE_FIELD_MASK = 'std_{}_error_rate'  # The format is std_{corruption}_error_rate.
TEST_ACCURACY_CLEAN_FIELD = stuff.wandb.TEST_ACCURACY_FIELD_MASK.format('clean')
TEST_ACCURACY_CORRUPTION_SEVERITY_FIELDS: str = None  # Late-initialization
TEST_ACCURACY_FIELD_PATTERN = re.compile(stuff.wandb.TEST_ACCURACY_FIELD_MASK.format('.*'))


def append_cifar10c_corruption_error_rates(df: DataFrame) -> DataFrame:
    # Compute metrics.
    clean_error_rate = 1 - df[TEST_ACCURACY_CLEAN_FIELD]
    mean_corruption_error_rates = pd.DataFrame(columns=CORRUPTION_FIELDS)
    std_corruption_error_rates = pd.DataFrame(columns=CORRUPTION_FIELDS)
    for curr_corruption_field in CORRUPTION_FIELDS:
        curr_corruption_severity_fields = list(filter(lambda arg: arg.startswith(f'{stuff.wandb.TEST_ACCURACY_FIELD_PREFIX}{curr_corruption_field}'), TEST_ACCURACY_CORRUPTION_SEVERITY_FIELDS))
        curr_corruption_error_rates = 1 - df[curr_corruption_severity_fields]
        curr_mean_corruption_error_rate = curr_corruption_error_rates.mean(axis='columns')
        mean_corruption_error_rates[curr_corruption_field] = curr_mean_corruption_error_rate
        std_corruption_error_rates[curr_corruption_field] = curr_corruption_error_rates.std(axis='columns')
    clean_error_rate.rename(CLEAN_ERROR_RATE_FIELD, inplace=True)
    mean_corruption_error_rates.rename(columns=dict(map(lambda arg: (arg, MEAN_CORRUPTION_ERROR_RATE_FIELD_MASK.format(arg)), CORRUPTION_FIELDS)), inplace=True)
    std_corruption_error_rates.rename(columns=dict(map(lambda arg: (arg, STD_CORRUPTION_ERROR_RATE_FIELD_MASK.format(arg)), CORRUPTION_FIELDS)), inplace=True)
    # Return computed values.
    return pd.concat((df, clean_error_rate, mean_corruption_error_rates, std_corruption_error_rates), axis='columns')


def append_cifar10c_corruption_errors(df: DataFrame, baseline_network: str) -> DataFrame:
    # Find the baseline.
    baseline_row = df[stuff.wandb.NETWORK_FIELD] == baseline_network
    # Compute metrics.
    clean_error_rate = df[CLEAN_ERROR_RATE_FIELD]
    corruption_error = pd.DataFrame(columns=CORRUPTION_FIELDS)
    relative_corruption_error = pd.DataFrame(columns=CORRUPTION_FIELDS)
    for curr_corruption_field in CORRUPTION_FIELDS:
        curr_corruption_severity_fields = list(filter(lambda arg: arg.startswith(f'{stuff.wandb.TEST_ACCURACY_FIELD_PREFIX}{curr_corruption_field}'), TEST_ACCURACY_CORRUPTION_SEVERITY_FIELDS))
        curr_corruption_error_rates = 1 - df[curr_corruption_severity_fields]
        curr_mean_corruption_error_rate = curr_corruption_error_rates.mean(axis='columns')
        corruption_error[curr_corruption_field] = curr_mean_corruption_error_rate / curr_mean_corruption_error_rate[baseline_row].item()
        relative_corruption_error[curr_corruption_field] = (curr_mean_corruption_error_rate - clean_error_rate) / (curr_mean_corruption_error_rate[baseline_row] - clean_error_rate[baseline_row]).item()
    mean_corruption_error = corruption_error.mean(axis='columns').rename(MEAN_CORRUPTION_ERROR_MASK.format(camel_to_snake(baseline_network)))
    mean_relative_corruption_error = relative_corruption_error.mean(axis='columns').rename(RELATIVE_MEAN_CORRUPTION_ERROR_MASK.format(camel_to_snake(baseline_network)))
    # Return computed values.
    return pd.concat((df, mean_corruption_error, mean_relative_corruption_error), axis='columns')


def camel_to_snake(camel: str) -> str:
    return re.sub(r'(?<!^)(?=[A-Z])', '_', camel).lower()


def snake_to_camel(snake: str) -> str:
    return ''.join([word.title() for word in snake.split('_')])


def save_latex_friendly_csv(df: DataFrame, path: str) -> None:
    df = df.rename(columns=dict(map(lambda snake: (snake, snake_to_camel(snake)), df.columns)))
    df.to_csv(path, index=False, na_rep='nan')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Weights & Biases arguments.
    group = parser.add_argument_group('logger arguments')
    group.add_argument('--wandb_entity', metavar='NAME', type=str, required=True, help='the name of the entity in the Weights & Biases framework')
    group.add_argument('--wandb_project', metavar='NAME', type=str, default=DEFAULT_WANDB_PROJECT, help='the name of the project in the Weights & Biases framework')
    # Result arguments.
    group = parser.add_argument_group('result arguments')
    group.add_argument('--output_dir', metavar='FILENAME', type=str, default=DEFAULT_OUTPUT, help='the path to the folder where resulting CSV will be written')
    # Parse arguments.
    args = parser.parse_args()
    # Call the magic procedure to download results from W&B.
    df = stuff.summarize_sweeps(entity_name=args.wandb_entity, project_name=args.wandb_project)
    datasets = df[[stuff.wandb.DATASET_FIELD]].drop_duplicates()
    networks = df[[stuff.wandb.NETWORK_FIELD]].drop_duplicates()
    TEST_ACCURACY_CORRUPTION_SEVERITY_FIELDS = list(filter(lambda arg: arg != TEST_ACCURACY_CLEAN_FIELD and bool(TEST_ACCURACY_FIELD_PATTERN.match(arg)), df.columns))
    CORRUPTION_FIELDS = list(sorted(set(map(lambda arg: arg[len(stuff.wandb.TEST_ACCURACY_FIELD_PREFIX):-2], TEST_ACCURACY_CORRUPTION_SEVERITY_FIELDS))))  # By construction, the format of the fields in TEST_ACCURACY_CORRUPTION_SEVERITY_FIELDS is '{prefix}{corruption}_{level}', where 1 <= level <= 5.
    # Save the original set of results.
    os.makedirs(args.output_dir, exist_ok=True)
    save_latex_friendly_csv(df, os.path.join(args.output_dir, 'sweep.csv'))
    # Save results by dataset.
    for _, (dataset,) in datasets.iterrows():
        save_latex_friendly_csv(df[df[stuff.wandb.DATASET_FIELD] == dataset], os.path.join(args.output_dir, f'sweep_{dataset}.csv'))
    # Save CIFAR-10-C error rate and corruption errors by network.
    df_cifar = df[df[stuff.wandb.DATASET_FIELD] == 'CIFAR10']
    df_cifar = append_cifar10c_corruption_error_rates(df_cifar)
    for _, (baseline_network,) in networks.iterrows():
        df_cifar = append_cifar10c_corruption_errors(df_cifar, baseline_network)
    save_latex_friendly_csv(df_cifar, os.path.join(args.output_dir, f'corruption_error_CIFAR10.csv'))
    for _, (network,) in networks.iterrows():
        values = df_cifar[df_cifar[stuff.wandb.NETWORK_FIELD] == network]
        rows = [OrderedDict([
            (stuff.wandb.NETWORK_FIELD, values[stuff.wandb.NETWORK_FIELD].item()),
            (stuff.wandb.DATASET_FIELD, values[stuff.wandb.DATASET_FIELD].item()),
            (CORRUPTION_FIELD, 'Clean'),
            (MEAN_CORRUPTION_ERROR_RATE_FIELD, values[CLEAN_ERROR_RATE_FIELD].item()),
            (STD_CORRUPTION_ERROR_RATE_FIELD, 0.0),
        ])]
        for corruption_field, mean_corruption_error_rate_field, std_corruption_error_rate_field in map(lambda arg: (arg, MEAN_CORRUPTION_ERROR_RATE_FIELD_MASK.format(arg), STD_CORRUPTION_ERROR_RATE_FIELD_MASK.format(arg)), CORRUPTION_FIELDS):
            corruption_severity_fields = list(filter(lambda arg: arg.startswith(f'{stuff.wandb.TEST_ACCURACY_FIELD_PREFIX}{corruption_field}'), TEST_ACCURACY_CORRUPTION_SEVERITY_FIELDS))
            severity_accuracy = values[corruption_severity_fields].to_numpy().flatten()
            p_value_fields = []
            for _, (other_network,) in networks.iterrows():
                if network != other_network:
                    other_values = df_cifar[df_cifar[stuff.wandb.NETWORK_FIELD] == other_network]
                    other_severity_accuracy = other_values[corruption_severity_fields].to_numpy().flatten()
                    p_value = stats.ttest_ind_from_stats(values[mean_corruption_error_rate_field].item(), values[std_corruption_error_rate_field].item(), len(corruption_severity_fields), other_values[mean_corruption_error_rate_field].item(), other_values[std_corruption_error_rate_field].item(), len(corruption_severity_fields), equal_var=False, alternative='two-sided').pvalue
                    p_value_fields.append((P_VALUE_FIELD_MASK.format(camel_to_snake(other_network)), p_value))
            rows.append(OrderedDict([
                (stuff.wandb.NETWORK_FIELD, values[stuff.wandb.NETWORK_FIELD].item()),
                (stuff.wandb.DATASET_FIELD, values[stuff.wandb.DATASET_FIELD].item()),
                (CORRUPTION_FIELD, ' '.join([word.title() for word in corruption_field.split('_')])),
                (MEAN_CORRUPTION_ERROR_RATE_FIELD, values[mean_corruption_error_rate_field].item()),
                (STD_CORRUPTION_ERROR_RATE_FIELD, values[std_corruption_error_rate_field].item()),
                *p_value_fields,
            ]))
        save_latex_friendly_csv(DataFrame(rows), os.path.join(args.output_dir, f'error_rate_CIFAR10_{network}.csv'))
