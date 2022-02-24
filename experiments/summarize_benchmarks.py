from benchmark import DEFAULT_WANDB_PROJECT
from summarize_sweeps import DEFAULT_OUTPUT, save_latex_friendly_csv
import argparse, os
import stuff


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
        save_latex_friendly_csv(df[df[stuff.wandb.NETWORK_FIELD] == network], os.path.join(args.output_dir, f'benchmark_{network}.csv'))
