import argparse
from emotenn import generate_data, plot, run, test, train


SUBMODULES = [generate_data, plot, run, test, train]


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    for submodule in SUBMODULES:
        try:
            main = getattr(submodule, 'main')
        except AttributeError:
            continue

        submodule_name = submodule.__name__.split('emotenn.')[1]
        subparser = subparsers.add_parser(submodule_name)
        subparser.set_defaults(main=main)

        try:
            fill_arguments_fn = getattr(submodule, 'fill_arguments')
            fill_arguments_fn(subparser)
        except AttributeError:
            pass

    return parser.parse_args()


def run_default():
    parser = argparse.ArgumentParser()
    run.fill_arguments(parser)
    run.main(parser.parse_args())


def main():
    args = parse_args()
    if hasattr(args, 'main'):
        args.main(args)
    else:
        run_default()


if __name__ == "__main__":
    main()
