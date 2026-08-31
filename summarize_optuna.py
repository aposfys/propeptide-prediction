'''
Aggregate a per-fold Optuna nested-CV run into one table.

Each outer fold is run as its own process (see run_optuna_gpu.sh), so each writes
its own fold_summary_outer{N}.json / best_params_outer{N}.json. This collects them
into the mean +/- std over the 5x4 = 20 models, and shows which hyperparameters
each fold picked.

Usage:
    python summarize_optuna.py --out_dir results/esm3_prop_optuna
'''
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from src.train_loop_crf import SEARCH_SPACES

# Reference points for the comparison this search is meant to settle. All are
# propeptide-specific F1 at +/-3 tolerance.
#
# The paper's headline figures are deliberately NOT listed here, for two reasons.
# They are precision and recall, not F1: "We find that DeepPeptide reaches a
# precision of 0.68 and a recall of 0.49 at a tolerance window of three AAs."
# And they cover peptides AND propeptides combined, so they would not be
# comparable to a propeptide-only model even after being reduced to an F1.
BASELINES = {
    'ESM3 propeptide-only, untuned (lr 1e-4)': 0.511,
    'ESM-2 propeptide-only, tuned T4 (lr 5.5e-3)': 0.626,
    'DeepPeptide paper, propeptides only (Fig. S7)': 0.535,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', required=True,
                   help='Directory the search wrote to (--out_dir of the training run).')
    args = p.parse_args()

    summary_files = sorted(glob.glob(os.path.join(args.out_dir, 'fold_summary_outer*.json')))
    if not summary_files:
        raise SystemExit(
            f'No fold_summary_outer*.json in {args.out_dir}. '
            'Either no fold has finished yet, or the run used --outer_fold=None '
            '(in which case read nested_cv_summary.json instead).'
        )

    per_model = []
    for f in summary_files:
        per_model.extend(json.load(open(f))['per_model'])

    df = pd.DataFrame(per_model)
    f1 = df['f1 propeptides']

    # Which search space produced these numbers decides what they may be compared to.
    space = 'table_s1'
    cfgs = sorted(glob.glob(os.path.join(args.out_dir, 'effective_config_outer*.json')))
    if cfgs:
        space = json.load(open(cfgs[0])).get('space', 'table_s1')

    print(f'Folds found : {len(summary_files)}/5  ({os.path.basename(args.out_dir)})')
    print(f'Models      : {len(df)}  (expected 4 per completed fold)')
    print(f'Search space: {space}')
    if space != 'table_s1':
        print('  WARNING: not the paper\'s Table S1 space. These numbers are exploratory')
        print('           and are NOT a like-for-like comparison against ESM-2 T4 (0.626),')
        print('           which was tuned over table_s1.')
    print()

    print('Per outer fold (mean over its 4 inner models):')
    by_fold = df.groupby('outer_fold')['f1 propeptides'].agg(['mean', 'std', 'count'])
    for fold, row in by_fold.iterrows():
        std = 0.0 if np.isnan(row['std']) else row['std']
        print(f'  outer {int(fold)} : F1 {row["mean"]:.4f} +/- {std:.4f}  (n={int(row["count"])})')
    print()

    # The paper reports "the average and standard deviations over 20 models on
    # their held-out test fold", so give mean +/- std for all three metrics.
    prec, rec = df['precision propeptides'], df['recall propeptides']
    print(f'Overall over {len(df)} models (propeptides, +/-3 tolerance):')
    print(f'  precision : {prec.mean():.4f} +/- {prec.std():.4f}')
    print(f'  recall    : {rec.mean():.4f} +/- {rec.std():.4f}')
    print(f'  F1        : {f1.mean():.4f} +/- {f1.std():.4f}   '
          f'(min {f1.min():.4f}, max {f1.max():.4f})')
    if len(df) == 4:
        print('  protocol  : standard CV, 1 held-out partition (paper\'s ablation setup)')
    elif len(df) == 20:
        print('  protocol  : full nested CV, 20 models (paper\'s headline setup)')
    print()

    print('Versus the reference numbers (propeptide F1, +/-3 tolerance):')
    for name, value in BASELINES.items():
        delta = f1.mean() - value
        print(f'  {name:<47} {value:.3f}   delta {delta:+.3f}')
    print('  (the two ESM rows are single-split runs; the paper row is nested CV,')
    print('   so expect some shift from protocol alone, independent of tuning)')
    print()

    # Which hyperparameters won, per fold — the actual deliverable of the search.
    best_files = sorted(glob.glob(os.path.join(args.out_dir, 'best_params_outer*.json')))
    if best_files:
        print('Winning hyperparameters per outer fold:')
        rows = {}
        for f in best_files:
            fold = os.path.basename(f).replace('best_params_outer', '').replace('.json', '')
            rows[f'outer {fold}'] = json.load(open(f))
        params_df = pd.DataFrame(rows).T
        with pd.option_context('display.width', 200, 'display.max_columns', 50):
            print(params_df.to_string())
        print()
        if 'lr' in params_df.columns:
            lrs = params_df['lr'].astype(float)
            print(f'lr chosen: min {lrs.min():.2e}  median {lrs.median():.2e}  max {lrs.max():.2e}')
            # If every fold's winner sits against a bound, the bound — not the data
            # — is deciding the answer, which is the signal to widen the space.
            lo, hi = SEARCH_SPACES.get(space, SEARCH_SPACES['table_s1'])['lr']
            if (lrs > hi / 1.25).all():
                print(f'  NOTE: every fold picked lr near the {hi:.0e} UPPER bound. The '
                      'bound is constraining the result — rerun with a wider space '
                      '(--space wide) before trusting these hyperparameters.')
            if (lrs < lo * 1.25).all():
                print(f'  NOTE: every fold picked lr near the {lo:.0e} LOWER bound. The '
                      'bound is constraining the result — widen the space downward '
                      'before trusting these hyperparameters.')
        print()

    out_csv = os.path.join(args.out_dir, 'summary_per_model.csv')
    df.to_csv(out_csv, index=False)
    print(f'Per-model metrics written to {out_csv}')

    # One fold is not an "incomplete" nested CV — it is the paper's ablation
    # protocol, run deliberately. Only flag a partial run when it is genuinely
    # a nested CV that has not finished.
    if len(summary_files) == 1:
        print()
        print('This is a single-partition (ablation) result — the paper\'s protocol for')
        print('comparing embedders. For the headline mean +/- std over 20 models, run the')
        print('other 4 outer folds and re-run this script.')
    elif len(summary_files) < 5:
        print()
        print(f'INCOMPLETE: {5 - len(summary_files)} outer fold(s) still missing — '
              'these numbers are partial.')


if __name__ == '__main__':
    main()
