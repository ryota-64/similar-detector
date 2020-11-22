import argparse
import os
import pathlib
import shutil



def main(args):
    root_dir = args.root_dir if args.root_dir else os.curdir()
    dynain_paths = pathlib.Path(root_dir).glob('**/dynain')
    for dynain_path in dynain_paths:
        shutil.copy(str(dynain_path), 'data/raw_data/20201023/dynain/{}_dynain'.format(dynain_path.parents[0].stem))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Optional arguments.
    parser.add_argument(
        '--root_dir',

    )

    args = parser.parse_args()
    main(args)
