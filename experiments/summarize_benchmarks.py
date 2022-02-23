from benchmark import DEFAULT_WANDB_PROJECT
import argparse, os
import stuff


DEFAULT_OUTPUT = 'results'


def _snake_to_camel(snake: str):
    return ''.join([i.title() for i in snake.split('_')])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Weights & Biases arguments.
    group = parser.add_argument_group('logger arguments')
    group.add_argument('--wandb_entity', metavar='NAME', type=str, required=True, help='the name of the entity in the Weights & Biases framework')
    group.add_argument('--wandb_project', metavar='NAME', type=str, default=DEFAULT_WANDB_PROJECT, help='the name of the project in the Weights & Biases framework')
    # Result arguments.
    group.add_argument('--output_dir', metavar='FILENAME', type=str, default=DEFAULT_OUTPUT, help='the name of the resulting CSV file')
    # Parse arguments.
    args = parser.parse_args()
    # Call the magic procedure and save to a CSV file.
    df = stuff.summarize_benchmarks(entity_name=args.wandb_entity, project_name=args.wandb_project)
    # Rename columns to LaTeX-friendly names.
    df.rename(columns=dict(map(lambda snake: (snake, _snake_to_camel(snake)), df.columns)), inplace=True)
    # Save complete set of results.
    os.makedirs(args.output_dir, exist_ok=True)
    df.to_csv(os.path.join(args.output_dir, 'benchmark_summary.csv'), index=False)
    # Save results by network.
    NAME_FIELD = _snake_to_camel(stuff.wandb.NAME_FIELD)
    names = df[[NAME_FIELD]].drop_duplicates()
    for _, (name,) in names.iterrows():
        df[(df[NAME_FIELD] == name)].to_csv(os.path.join(args.output_dir, f'benchmark_{name}.csv'), index=False)
