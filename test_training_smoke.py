'''
Run the actual training loop for one epoch, at both grammar sizes.

test_grammar.py and test_architecture.py check the model and the label encoder.
Neither executes `train()`, and that gap let a real bug reach the GPU: a
module-level helper named `score` was shadowed by a local `score = ...` inside
`run_training_for_params`, so the call above that assignment raised
UnboundLocalError. Python resolves that at runtime, per function, so no import,
no syntax check and no unit test on the model could see it. Only running the
loop could.

This builds a tiny synthetic embedding directory from the real data files and
runs one epoch on CPU, at 51 and at 101 states. It is a smoke test, not a
correctness test: it asserts the loop completes, writes the files a run is
supposed to write, and records the right grammar.

    python test_training_smoke.py

Takes a couple of minutes on CPU. No GPU, no real embeddings, no network.
'''
import argparse
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

EMBEDDING_DIM = 32          # tiny on purpose; the loop does not care
failures = []


def check(condition, message):
    print(('  PASS  ' if condition else '  FAIL  ') + message)
    if not condition:
        failures.append(message)


def build_fake_embeddings(data_file, partitioning_file, out_dir, limit=60):
    '''One small random tensor per sequence, keyed by md5 exactly as the real
    extractors key them, for a handful of proteins from each partition.'''
    from src.utils.dataset import make_hashes

    frame = pd.read_csv(data_file).fillna('')
    partitioning = pd.read_csv(partitioning_file)
    frame = frame.merge(partitioning, left_on='protein_id', right_on='AC')

    keep = []
    for cluster in sorted(frame['cluster'].unique()):
        subset = frame[frame['cluster'] == cluster]
        # Short sequences only, so one CPU epoch is quick.
        subset = subset[subset['sequence'].str.len() < 250]
        keep.append(subset.head(limit))
    frame = pd.concat(keep, ignore_index=True)

    os.makedirs(out_dir, exist_ok=True)
    for digest, sequence in zip(make_hashes(frame['sequence'].tolist()),
                                frame['sequence'].tolist()):
        path = os.path.join(out_dir, f'{digest}.pt')
        if not os.path.isfile(path):
            torch.save(torch.randn(len(sequence), EMBEDDING_DIM), path)
    return frame


def main():
    sys.path.insert(0, os.getcwd())
    from src.train_loop_crf import train

    workspace = tempfile.mkdtemp(prefix='smoke_')
    embeddings = os.path.join(workspace, 'emb')
    frame = build_fake_embeddings('data/labeled_sequences.csv',
                                  'data/graphpart_assignments.csv', embeddings)

    # Write the trimmed data and partition files the run will actually read.
    data_file = os.path.join(workspace, 'labeled.csv')
    partition_file = os.path.join(workspace, 'partitions.csv')
    frame.drop(columns=['AC', 'cluster']).to_csv(data_file, index=False)
    frame[['AC', 'cluster']].rename(columns={'AC': 'AC'}).to_csv(partition_file, index=False)
    print(f'{len(frame)} proteins, {EMBEDDING_DIM}-dim random embeddings, {workspace}\n')

    for max_len in (50, 100):
        out_dir = os.path.join(workspace, f'run_{max_len + 1}')
        args = argparse.Namespace(
            embeddings_dir=embeddings, data_file=data_file,
            partitioning_file=partition_file, embedding='precomputed',
            embedding_dim=EMBEDDING_DIM, model='lstmcnncrf', out_dir=out_dir,
            epochs=1, patience=0, batch_size=8, lr=1e-3, dropout=0.1,
            conv_dropout=0.1, kernel_size=3, num_filters=8, hidden_size=8,
            weight_decay=0.0, seed=42, use_focal=False, allow_cpu=True,
            num_workers=0, num_cpu_threads=1,
            max_peptide_len=max_len, min_peptide_len=5,
        )
        os.makedirs(out_dir, exist_ok=True)
        print(f'--- one epoch at {max_len + 1} states ---')
        try:
            best_val, test_metrics = train(args)
            ok = True
        except Exception as exc:
            ok = False
            print(f'    {type(exc).__name__}: {exc}')
        check(ok, f'{max_len + 1}-state training loop completes an epoch')
        if not ok:
            continue

        check(os.path.isfile(os.path.join(out_dir, 'test_metrics.json')),
              f'{max_len + 1}: test_metrics.json written')
        check(os.path.isfile(os.path.join(out_dir, 'valid_metrics.json')),
              f'{max_len + 1}: valid_metrics.json written')
        check(os.path.isfile(os.path.join(out_dir, 'test_outputs.pickle')),
              f'{max_len + 1}: test_outputs.pickle written')

        written = json.load(open(os.path.join(out_dir, 'test_metrics.json')))
        check('f1 propeptides@1' in written and 'f1 propeptides@3' in written,
              f'{max_len + 1}: both tolerances recorded')
        check(abs(written['f1 propeptides'] - written['f1 propeptides@3']) < 1e-12,
              f'{max_len + 1}: the bare key still holds the +/-3 value')
        check(isinstance(best_val, dict) and 'f1 propeptides' in best_val,
              f'{max_len + 1}: validation metrics returned')

    shutil.rmtree(workspace, ignore_errors=True)
    print()
    print(f'{len(failures)} failure(s)')
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
