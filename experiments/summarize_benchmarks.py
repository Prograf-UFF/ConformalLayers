from benchmark import DEFAULT_WANDB_PROJECT
from summarize_sweeps import DEFAULT_OUTPUT, snake_to_camel
import argparse, os
import stuff


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
    # Call the magic procedure and save to a CSV file.
    df = stuff.summarize_benchmarks(entity_name=args.wandb_entity, project_name=args.wandb_project)
    # Rename columns to LaTeX-friendly names.
    df.rename(columns=dict(map(lambda snake: (snake, snake_to_camel(snake)), df.columns)), inplace=True)
    # Save the complete set of results.
    os.makedirs(args.output_dir, exist_ok=True)
    df.to_csv(os.path.join(args.output_dir, 'benchmark.csv'), index=False)
    # Save results by network.
    NETWORK_FIELD = snake_to_camel(stuff.wandb.NETWORK_FIELD)
    networks = df[[NETWORK_FIELD]].drop_duplicates()
    for _, (network,) in networks.iterrows():
        df[(df[NETWORK_FIELD] == network)].to_csv(os.path.join(args.output_dir, f'benchmark_{network}.csv'), index=False)
