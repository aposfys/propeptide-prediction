'''
Simple 5-fold CV — one job per fold, fixed hyperparameters, no Optuna.

Matches the original DeepPeptide *code* (fteufel/DeepPeptide, run.py), which is
just `train(parse_arguments())`: the training loop, with hyperparameters read
from the command line. Upstream ships no hyperparameter-search code at all.

This is NOT the paper's protocol. That is a 5-fold nested CV with an Optuna
search over Table S1's space in the inner loop, reimplemented from the paper's
description in objective(), src/train_loop_crf.py — run it via run_optuna_gpu.sh.

Use this script for smoke tests, and for reproducing a fixed configuration
without re-running a search: Table S2 of btad616 publishes the per-fold winners
T0..T4 that the authors' own search found.

Works on every branch (uses train() which is present everywhere).
Gracefully handles branches where parse_arguments() lacks some flags.

Table S2 (btad616) — hyperparameters found by the paper's search:
    fold   lr       batch  dropout  conv_dropout  kernel  filters  hidden
    T0     0.0033   90     0.2349   0.1041        5       96       48
    T1     0.0003   70     0.3255   0.3928        5       80       32
    T2     0.0010   50     0.1437   0.4085        3       80       32
    T3     0.0033   60     0.4781   0.5082        5       96       32
    T4     0.0055   20     0.6902   0.2672        5       48       48

Usage (fold 0, with Table S2's T0 row):
    python run_fold.py --test_fold 0 \
        --embeddings_dir ~/embeddings/esm2 \
        --data_file data/labeled_sequences.csv \
        --partitioning_file data/graphpart_assignments.csv \
        --embedding_dim 1280 \
        --lr 0.0033 --batch_size 90 --dropout 0.2349 --conv_dropout 0.1041 \
        --kernel_size 5 --num_filters 96 --hidden_size 48 \
        --out_dir results/s2_esm2
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
