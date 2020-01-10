import argparse
import importlib


SUBMODULES = ['generate_data', 'plot', 'run', 'test', 'train']


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    for submodule_name in SUBMODULES:
        submodule = importlib.import_module(f'emotenn.{submodule_name}')
        try:
            main = getattr(submodule, 'main')
        except AttributeError:
            continue

        subparser = subparsers.add_parser(submodule_name)
        subparser.set_defaults(main=main)

        try:
            fill_arguments_fn = getattr(submodule, 'fill_arguments')
            fill_arguments_fn(subparser)
        except AttributeError:
            pass

    return parser.parse_args()


def main():
    args = parse_args()
    print(args)
    args.main(args)


if __name__ == "__main__":
    main()
