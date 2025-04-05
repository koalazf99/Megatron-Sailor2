import os
import sys
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from megatron.data import indexed_dataset
from math import ceil


def merge_shard(file_group, output_prefix):
    builder = None
    for prefix in sorted(file_group):
        if builder is None:
            dataset = indexed_dataset.make_dataset(prefix, 'infer')
            if isinstance(dataset, indexed_dataset.MMapIndexedDataset):
                builder = indexed_dataset.MMapIndexedDatasetBuilder(output_prefix + '.bin', dtype=dataset._index.dtype)
            else:
                builder = indexed_dataset.IndexedDatasetBuilder(output_prefix + '.bin')
            del dataset
        builder.merge_file_(prefix)
    builder.finalize(output_prefix + '.idx')


def main(args):
    # Collect valid prefixes
    all_files = os.listdir(args.input)
    prefix_set = set()

    for basename in all_files:
        prefix, ext = os.path.splitext(basename)
        if prefix in prefix_set:
            continue
        if not os.path.isfile(os.path.join(args.input, basename)):
            continue

        other_ext = '.bin' if ext == '.idx' else '.idx'
        full_path = os.path.join(args.input, prefix)
        assert os.path.isfile(full_path + other_ext), f'Missing counterpart file for {full_path}'

        prefix_set.add(prefix)

    sorted_prefixes = sorted(list(prefix_set))
    total_shards = ceil(len(sorted_prefixes) / args.files_per_shard)

    print(f"Total files: {len(sorted_prefixes)}; Merging into {total_shards} shards...")

    for shard_id in range(total_shards):
        start = shard_id * args.files_per_shard
        end = min((shard_id + 1) * args.files_per_shard, len(sorted_prefixes))
        shard_prefixes = [os.path.join(args.input, p) for p in sorted_prefixes[start:end]]

        output_path = f"{args.output_prefix}_{shard_id:04d}"
        print(f"  → Merging files {start} to {end-1} into {output_path}")
        merge_shard(shard_prefixes, output_path)

    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True,
                        help='Path to directory containing dataset files to merge')
    parser.add_argument('--output_prefix', type=str, required=True,
                        help='Prefix for output merged files (shard index will be appended)')
    parser.add_argument('--files_per_shard', type=int, required=True,
                        help='Number of .bin+.idx file pairs to merge per shard')

    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_prefix)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    assert os.path.isdir(args.input), f'Input path {args.input} is not a directory or does not exist'
    assert os.path.isdir(os.path.dirname(args.output_prefix)), f'Directory {os.path.dirname(args.output_prefix)} does not exist'

    main(args)
