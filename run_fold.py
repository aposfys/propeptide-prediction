'''
Simple 5-fold CV — one job per fold, fixed hyperparameters, no Optuna.
Matches the original DeepPeptide paper training protocol.

Works on every branch (uses train() which is present everywhere).
Gracefully handles branches where parse_arguments() lacks some flags.

Usage (one fold):
    python run_fold.py --test_fold 0 \
        --embeddings_dir ~/embeddings/esm2 \
        --data_file data/labeled_sequences.csv \
        --partitioning_file data/graphpart_assignments.csv \
        --embedding_dim 1280 \
        --out_dir results/main_esm2

Run all 5 in parallel via:
    THREADS=8 bash run_simple_cv.sh --embeddings_dir ... --out_dir ...
'''
import argparse
import json
import os
import sys

import torch

# Partition scheme: test=fold, val=(fold+1)%5, train=remaining 3
_PARTITIONS = {
    0: dict(train=[2, 3, 4], val=[1], test=[0]),
    1: dict(train=[0, 3, 4], val=[2], test=[1]),
    2: dict(train=[0, 1, 4], val=[3], test=[2]),
    3: dict(train=[0, 1, 2], val=[4], test=[3]),
    4: dict(train=[1, 2, 3], val=[0], test=[4]),
}

# Args handled here, stripped from sys.argv before branch's parse_arguments() sees them.
_LOCAL_ARGS = {'--test_fold', '--num_cpu_threads', '--num_workers'}


def _pop_local_args(argv):
    '''Extract our local args; return (values_dict, remaining_argv).'''
    values = {'test_fold': None, 'num_cpu_threads': None, 'num_workers': 0}
    clean = []
    i = 0
    while i < len(argv):
        consumed = False
        for key in _LOCAL_ARGS:
            flag = key.lstrip('-').replace('-', '_')
            if argv[i] == key and i + 1 < len(argv):
                values[flag] = argv[i + 1]
                i += 2
                consumed = True
                break
            elif argv[i].startswith(f'{key}='):
                values[flag] = argv[i].split('=', 1)[1]
                i += 1
                consumed = True
                break
        if not consumed:
            clean.append(argv[i])
            i += 1
    return values, clean


def main():
    local, remaining = _pop_local_args(sys.argv[1:])

    if local['test_fold'] is None:
        print('ERROR: --test_fold {0..4} is required.', file=sys.stderr)
        sys.exit(1)

    test_fold = int(local['test_fold'])
    num_threads = int(local['num_cpu_threads']) if local['num_cpu_threads'] else None
    num_workers = int(local['num_workers']) if local['num_workers'] else 0

    if num_threads:
        torch.set_num_threads(num_threads)
        print(f'PyTorch CPU threads: {num_threads}', flush=True)

    # Let the branch's own parse_arguments() handle everything else
    sys.argv = [sys.argv[0]] + remaining
    from src.train_loop_crf import parse_arguments, train
    args = parse_arguments()

    # Patch in values that branches may not expose as CLI flags
    if not hasattr(args, 'num_workers'):
        args.num_workers = num_workers
    if not hasattr(args, 'num_cpu_threads'):
        args.num_cpu_threads = num_threads

    p = _PARTITIONS[test_fold]
    print(
        f'Fold {test_fold}: train={p["train"]}  val={p["val"]}  test={p["test"]}',
        flush=True,
    )

    # Each fold writes to its own subdir so 5 parallel jobs never collide
    base_out = args.out_dir
    args.out_dir = os.path.join(base_out, f'fold{test_fold}')
    os.makedirs(args.out_dir, exist_ok=True)

    best_val, test_metrics = train(
        args,
        train_partitions=p['train'],
        valid_partitions=p['val'],
        test_partitions=p['test'],
    )

    summary = {'fold': test_fold, 'val': best_val, 'test': test_metrics}
    out_path = os.path.join(base_out, f'summary_fold{test_fold}.json')
    json.dump(summary, open(out_path, 'w'), indent=2)

    t = test_metrics or {}
    print(
        f'\nFold {test_fold} done:  '
        f"pep F1={t.get('f1 peptides', 0):.3f}  "
        f"pro F1={t.get('f1 propeptides', 0):.3f}  "
        f"all F1={t.get('f1 all', 0):.3f}",
        flush=True,
    )


if __name__ == '__main__':
    main()
