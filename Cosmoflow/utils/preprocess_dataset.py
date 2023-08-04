"""Split original CosmoFlow dataset into training samples."""

# Some code adapted from:
# https://github.com/mlcommons/hpc/blob/main/cosmoflow/prepare.py

import argparse
import os
import os.path
import glob
import multiprocessing
import functools
import math
import pickle

import numpy as np
import h5py
import tqdm


parser = argparse.ArgumentParser(
    description='Split CosmoFlow dataset for training')
parser.add_argument('data_dir', type=str,
                    help='Directory containing CosmoFlow dataset')
parser.add_argument('out_dir', type=str,
                    help='Directory to output data')
parser.add_argument('--num-train', type=int, default=14,
                    help='Number of training universes (default: 4096)')
parser.add_argument('--num-val', type=int, default=2,
                    help='Number of validation universes (default: 1024)')
parser.add_argument('--num-test', type=int, default=0,
                    help='Number of test samples (default: 1024)')
parser.add_argument('--train-list', type=str, default=None,
                    help='File with list of universes for training')
parser.add_argument('--val-list', type=str, default=None,
                    help='File with list of universes for validation')
parser.add_argument('--test-list', type=str, default=None,
                    help='File with list of universes for testing')
parser.add_argument('--no-transpose', action='store_true',
                    help='Do not transpose data')
parser.add_argument('--ntasks', type=int, default=8,
                    help='Number of tasks to use for processing (default: 8)')
parser.add_argument('--split-size', type=int, default=128,
                    help='Size of each universe split (default: 128)')
parser.add_argument('--unis-per-dir', type=int, default=64,
                    help='Universes per output subdirectory (default: 64)')


def list_files(data_dir):
    """Recursively list all HDF5 files in data_dir."""
    return glob.glob(data_dir + '/**/*.hdf5', recursive=True)


def make_output_dir(out_dir, num_subdirs):
    """Make output directory hierarchy."""
    os.makedirs(out_dir)
    for i in range(num_subdirs):
        os.makedirs(os.path.join(out_dir, f'{i:03d}'))


def make_index_file(out_dir, num_subdirs, split_size, files):
    """Write out an index file.

    This lists some basic info about the dataset to avoid having to
    rescan the whole directory structure.

    """
    data = {
        'split_size': split_size,
        'num_subdirs': num_subdirs,
        'filenames': list(
            map(lambda x: os.path.splitext(os.path.basename(x))[0], files))
    }
    with open(os.path.join(out_dir, 'idx'), 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def read_file(filename):
    """Load universe and target from filename."""
    with h5py.File(filename, 'r') as f:
        return f['full'][:], f['unitPar'][:]


def write_file(filename, x, y):
    """Write x and y to filename."""
    with h5py.File(filename, 'w') as f:
        f.create_dataset('split', data=x)
        f.create_dataset('unitPar', data=y)


def split_universe(x, split_size):
    """Yield sub-cubes of the universe."""
    num_splits = x.shape[0] // split_size
    for xi in np.split(x, num_splits, axis=0):
        for xij in np.split(xi, num_splits, axis=1):
            for xijk in np.split(xij, num_splits, axis=2):
                yield xijk


def process_file(idx_filename, out_dir, split_size, unis_per_dir, transpose):
    """Extract splits from filename and write them out."""
    idx, filename = idx_filename
    x, y = read_file(filename)
    base_outname = os.path.join(
        out_dir,
        f'{idx // unis_per_dir:03d}',
        os.path.splitext(os.path.basename(filename))[0])
    for i, split in enumerate(split_universe(x, split_size)):
        if transpose:
            # Swap from HWDC to CHWD.
            split = split.transpose((3, 0, 1, 2))
        write_file(base_outname + f'_{i:03d}.hdf5', split, y)


def process_file_chunk(args, chunk_name, files):
    """Split a set of universes for a particular chunk."""
    print(f'Processing chunk {chunk_name} with {len(files)} universes')
    out_dir = os.path.join(args.out_dir, chunk_name)
    make_output_dir(out_dir, int(math.ceil(len(files) / args.unis_per_dir)))
    make_index_file(out_dir, args.unis_per_dir, args.split_size, files)
    with multiprocessing.Pool(processes=args.ntasks) as pool:
        for _ in tqdm.tqdm(pool.imap(functools.partial(
                process_file, out_dir=out_dir,
                split_size=args.split_size, unis_per_dir=args.unis_per_dir,
                transpose=not args.no_transpose),
                                     enumerate(files)),
                           total=len(files)):
            pass


def load_univs_list(filename):
    """Return a set of universes in filename."""
    with open(filename, 'r') as f:
        return set(map(lambda x: x.strip(), f.readlines()))


def get_filesets(args, files):
    """Return the sets of training/validation/test/extra files."""
    # Require either none or all of the lists are given.
    listcnt = [bool(args.train_list),
               bool(args.val_list),
               bool(args.test_list)]
    if sum(listcnt) == 0:
        # Use counts.
        files = sorted(files)
        start, end = 0, args.num_train
        train = files[start:end]
        start = end
        end += args.num_val
        val = files[start:end]
        start = end
        end += args.num_test
        test = files[start:end]
        extra = files[end:]
        return train, val, test, extra
    if sum(listcnt) == 3:
        # Use lists.
        train_univs = load_univs_list(args.train_list)
        val_univs = load_univs_list(args.val_list)
        test_univs = load_univs_list(args.test_list)

        def get_univ(name):
            return os.path.splitext(os.path.basename(name))[0]

        # This is a bit of a hack to deal with duplicated universe names.
        def dedup_univ(filenames):
            seen = set()
            dedup = []
            for filename in filenames:
                univ = get_univ(filename)
                if univ not in seen:
                    seen.add(univ)
                    dedup.append(filename)
            return dedup

        train = list(filter(lambda x: get_univ(x) in train_univs, files))
        train = dedup_univ(train)
        if len(train) != len(train_univs):
            raise RuntimeError('Missing training universes')
        val = list(filter(lambda x: get_univ(x) in val_univs, files))
        val = dedup_univ(val)
        if len(val) != len(val_univs):
            raise RuntimeError('Missing validation universes')
        test = list(filter(lambda x: get_univ(x) in test_univs, files))
        test = dedup_univ(test)
        if len(test) != len(test_univs):
            raise RuntimeError('Missing testing universes')
        extra = list(set(files) - set(train) - set(val) - set(test))
        extra = dedup_univ(extra)
        return train, val, test, extra
    raise ValueError('Must specify all lists if using lists')


def process_all_files(args):
    """Split all files per args."""
    if not os.path.isdir(args.data_dir):
        raise ValueError(f'Bad data-dir: {args.data_dir}')
    files = list_files(args.data_dir)
    train, val, test, extra = get_filesets(args, files)
    process_file_chunk(args, 'train', train)
    process_file_chunk(args, 'validation', val)
    process_file_chunk(args, 'test', test)
    if extra:
        process_file_chunk(args, 'extra', extra)


if __name__ == '__main__':
    process_all_files(parser.parse_args())
