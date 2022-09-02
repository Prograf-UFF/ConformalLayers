from typing import OrderedDict
from benchmark import DEFAULT_WANDB_PROJECT
from scipy import stats
from summarize_sweeps import DEFAULT_OUTPUT, camel_to_snake, save_latex_friendly_csv, snake_to_camel
import argparse, os
import pandas as pd
import stuff


P_VALUE_ELAPSED_TIME_FIELD_MASK = 'p_value_elapsed_time_{}'  # The format is p_value_elapsed_time_{compared-network}
P_VALUE_MAX_MEMORY_FIELD_MASK = 'p_value_max_memory_{}'  # The format is p_value_max_memory_{compared-network}
P_VALUE_EMISSION_FIELD_MASK = 'p_value_emission_{}'  # The format is p_value_emission_{compared-network}


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
    df = stuff.summarize_benchmarks(entity_name=args.wandb_entity, project_name=args.wandb_project)
    # Save the complete set of results.
    os.makedirs(args.output_dir, exist_ok=True)
    save_latex_friendly_csv(df, os.path.join(args.output_dir, 'benchmark.csv'))
    # Save results by network.
    networks = df[[stuff.wandb.NETWORK_FIELD]].drop_duplicates()
    for _, (network,) in networks.iterrows():
        values = df[df[stuff.wandb.NETWORK_FIELD] == network]
        p_value_elapsed_time_fields = []
        p_value_max_memory_fields = []
        p_value_emission_fields = []
        for _, (other_network,) in networks.iterrows():
            if network != other_network:
                other_values = df[df[stuff.wandb.NETWORK_FIELD] == other_network]
                p_value = stats.ttest_ind_from_stats(values[stuff.wandb.MEAN_ELAPSED_TIME_FIELD].to_numpy(), values[stuff.wandb.STD_ELAPSED_TIME_FIELD].to_numpy(), values[stuff.wandb.NUM_SAMPLES_FIELD].to_numpy(), other_values[stuff.wandb.MEAN_ELAPSED_TIME_FIELD].to_numpy(), other_values[stuff.wandb.STD_ELAPSED_TIME_FIELD].to_numpy(), other_values[stuff.wandb.NUM_SAMPLES_FIELD].to_numpy(), equal_var=False, alternative='two-sided').pvalue
                p_value_elapsed_time_fields.append((P_VALUE_ELAPSED_TIME_FIELD_MASK.format(camel_to_snake(other_network)), p_value))
                p_value = stats.ttest_ind_from_stats(values[stuff.wandb.MEAN_MAX_MEMORY_FIELD].to_numpy(), values[stuff.wandb.STD_MAX_MEMORY_FIELD].to_numpy(), values[stuff.wandb.NUM_SAMPLES_FIELD].to_numpy(), other_values[stuff.wandb.MEAN_MAX_MEMORY_FIELD].to_numpy(), other_values[stuff.wandb.STD_MAX_MEMORY_FIELD].to_numpy(), other_values[stuff.wandb.NUM_SAMPLES_FIELD].to_numpy(), equal_var=False, alternative='two-sided').pvalue
                p_value_max_memory_fields.append((P_VALUE_MAX_MEMORY_FIELD_MASK.format(camel_to_snake(other_network)), p_value))
                p_value = stats.ttest_ind_from_stats(values[stuff.wandb.MEAN_EMISSION_FIELD].to_numpy(), values[stuff.wandb.STD_EMISSION_FIELD].to_numpy(), values[stuff.wandb.NUM_SAMPLES_FIELD].to_numpy(), other_values[stuff.wandb.MEAN_EMISSION_FIELD].to_numpy(), other_values[stuff.wandb.STD_EMISSION_FIELD].to_numpy(), other_values[stuff.wandb.NUM_SAMPLES_FIELD].to_numpy(), equal_var=False, alternative='two-sided').pvalue
                p_value_emission_fields.append((P_VALUE_EMISSION_FIELD_MASK.format(camel_to_snake(other_network)), p_value))
        p_value_elapsed_time_fields = pd.DataFrame(OrderedDict(p_value_elapsed_time_fields), index=values.index)
        p_value_max_memory_fields = pd.DataFrame(OrderedDict(p_value_max_memory_fields), index=values.index)
        p_value_emission_fields = pd.DataFrame(OrderedDict(p_value_emission_fields), index=values.index)
        save_latex_friendly_csv(pd.concat((values, p_value_elapsed_time_fields, p_value_max_memory_fields, p_value_emission_fields), axis='columns'), os.path.join(args.output_dir, f'benchmark_{network}.csv'))
