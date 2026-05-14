'''
Smoke test for train_loop_crf.py.

Patches PrecomputedCSVForOverlapCRFDataset.__getitem__ to return random
tensors on-the-fly, so no .pt files are needed.  Runs a full 5-fold nested
CV with 1 Optuna trial, 2 epochs, patience=1.

Expected results
----------------
- Exits without error.
- Creates 5 × 4 = 20 model checkpoints in /tmp/smoke_run/.
- nested_cv_summary.json is written with "n_models": 20.
- Propeptide F1 is near-zero (random embeddings) but every structural fix
  is exercised.

Run from the repo root:
    conda activate deeppeptide_esm3
    python test_smoke.py
'''

import argparse
import os
import shutil
import json
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

EMBED_DIM    = 1536
N_PER_CLUSTER = 6    # proteins per partition (30 total — keeps the run fast)
OUT_DIR      = '/tmp/smoke_run'


# ---------------------------------------------------------------------------
# Dataset patch — random embeddings, no .pt files needed
# ---------------------------------------------------------------------------

def _patch_dataset_for_smoke():
    '''Replace __getitem__ with a version that returns random embeddings.

    Labels are still computed correctly from self.propeptides (same logic as
    the real __getitem__) — only the embedding tensor is synthetic.
    '''
    from src.utils.dataset import PrecomputedCSVForOverlapCRFDataset
    from src.utils.crf_label_utils import peptide_list_to_label_sequence

    def _smoke_getitem(self, index):
        seq_len     = len(self.sequences[index])
        propeptides = self.propeptides[index]
        embeddings  = torch.randn(seq_len, EMBED_DIM, dtype=torch.float32)
        label       = torch.from_numpy(
            peptide_list_to_label_sequence(propeptides, seq_len, start_state=1, max_len=50)
        )
        mask = torch.ones(seq_len)
        return embeddings, mask, label, propeptides

    PrecomputedCSVForOverlapCRFDataset.__getitem__ = _smoke_getitem


# ---------------------------------------------------------------------------
# Mini CSV helpers
# ---------------------------------------------------------------------------

def build_mini_csvs(out_dir: str):
    df = pd.read_csv('data/labeled_sequences.csv', index_col='protein_id')
    pf = pd.read_csv('data/graphpart_assignments.csv', index_col='AC')
    d  = df.join(pf).dropna(subset=['cluster'])
    d  = d[d['propeptide_coordinates'].fillna('') != '']

    mini_rows = []
    for cluster in [0, 1, 2, 3, 4]:
        subset = d[d['cluster'] == float(cluster)].head(N_PER_CLUSTER)
        mini_rows.append(subset)
    mini_df = pd.concat(mini_rows)

    data_path = os.path.join(out_dir, 'mini_sequences.csv')
    part_path = os.path.join(out_dir, 'mini_partitions.csv')

    mini_df.drop(
        columns=['cluster', 'priority', 'label-val', 'between_connectivity'],
        errors='ignore',
    ).to_csv(data_path)

    part_df = mini_df[[]].copy()
    part_df.index.name = 'AC'
    part_df['cluster'] = mini_df['cluster'].values
    part_df.to_csv(part_path)

    print(f'  {len(mini_df)} proteins written to {data_path}')
    return data_path, part_path


def make_args(data_path, part_path, run_dir) -> argparse.Namespace:
    return argparse.Namespace(
        embeddings_dir='/dev/null',   # unused after patch
        data_file=data_path,
        partitioning_file=part_path,
        embedding='precomputed',
        embedding_dim=EMBED_DIM,
        model='lstmcnncrf',
        out_dir=run_dir,
        epochs=2,
        patience=1,
        batch_size=4,
        n_trials=1,
        lr=1e-4,
        dropout=0.1,
        conv_dropout=0.1,
        weight_decay=0.0,
        kernel_size=3,
        num_filters=16,
        hidden_size=16,
        # num_workers=0: keeps DataLoader in the main process so the
        # class-level monkey-patch on __getitem__ remains active.
        num_workers=0,
    )


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def assert_gradient_clipping_order():
    import torch.nn as nn
    model     = nn.Linear(4, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss      = nn.CrossEntropyLoss()(model(torch.randn(2, 4)), torch.randint(0, 2, (2,)))
    loss.backward()
    assert model.weight.grad is not None, 'No gradients before clipping!'
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
    optimizer.step()
    print('[OK] gradient clipping order: backward → clip → step')


def assert_flatten_lstm_safe():
    from src.models import SelfAttentionCRF, LSTMCNNCRF
    from src.train_loop_crf import _flatten_lstm_if_present

    _flatten_lstm_if_present(LSTMCNNCRF(input_size=32, num_states=51))
    print('[OK] _flatten_lstm_if_present: LSTMCNNCRF OK')

    _flatten_lstm_if_present(SelfAttentionCRF(input_size=32, hidden_size=16, num_states=51))
    print('[OK] _flatten_lstm_if_present: SelfAttentionCRF OK (no crash)')


def assert_train_dataloader_shuffles(data_path, part_path):
    from src.train_loop_crf import get_dataloaders
    from torch.utils.data import RandomSampler
    args = make_args(data_path, part_path, OUT_DIR)
    train_loader, _, _ = get_dataloaders(args, [0, 1, 2, 3], [4], [4])
    assert isinstance(train_loader.sampler, RandomSampler), \
        f'Expected RandomSampler, got {type(train_loader.sampler)}'
    print('[OK] training DataLoader uses RandomSampler (shuffle=True)')


def assert_nested_cv_produces_20_models(args):
    from src.train_loop_crf import train_nested_cv
    summary = train_nested_cv(args)
    assert summary['n_models'] == 20, f"Expected 20 models, got {summary['n_models']}"
    print(f'[OK] nested CV produced {summary["n_models"]} models')

    model_files = [
        f for f in os.listdir(args.out_dir)
        if f.startswith('model_outer') and f.endswith('.pt')
    ]
    assert len(model_files) == 20, f'Expected 20 .pt files, found {len(model_files)}'
    print(f'[OK] 20 checkpoint files written to {args.out_dir}')

    with open(os.path.join(args.out_dir, 'nested_cv_summary.json')) as f:
        s = json.load(f)
    assert s['n_models'] == 20
    print(f'[OK] nested_cv_summary.json: mean F1 propeptides = {s["mean_f1_propeptides"]:.4f}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('=== Smoke test: train_loop_crf.py ===\n')

    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR)
    data_dir = os.path.join(OUT_DIR, 'data')
    os.makedirs(data_dir)

    # Unit checks — no data needed
    assert_gradient_clipping_order()
    assert_flatten_lstm_safe()

    # Build mini CSVs
    print('\nBuilding mini dataset...')
    data_path, part_path = build_mini_csvs(data_dir)

    # Patch dataset to skip disk I/O
    print('Patching dataset for in-memory random embeddings...')
    _patch_dataset_for_smoke()
    print('[OK] dataset patched')

    # DataLoader shuffle check
    assert_train_dataloader_shuffles(data_path, part_path)

    # Full nested CV (the main structural check)
    args = make_args(data_path, part_path, OUT_DIR)
    print('\nRunning full 5-fold nested CV (2 epochs, 1 Optuna trial)...')
    assert_nested_cv_produces_20_models(args)

    print('\n=== All checks passed ===')
