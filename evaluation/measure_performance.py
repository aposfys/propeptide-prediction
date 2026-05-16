'''
Compute performance metrics for the propeptide-only ESM3 nested-CV pipeline.

Reads the test_outputs_outer{i}_inner{j}.pickle files saved by train_loop_crf.py
and reports precision / recall / F1 for propeptide detection at tolerances 0–3.

Usage (from repo root):
    python evaluation/measure_performance.py \
        --out_dir /data/apostolos/nested_cv_run \
        --data_file data/labeled_sequences.csv
'''

import argparse
import os
import pickle
from typing import List, Tuple

import pandas as pd
from tqdm.auto import tqdm

# ---- CRF state constants (propeptide-only model: 51 states, 0=background) ----
PROPEPTIDE_START_STATE = 1
PROPEPTIDE_END_STATE   = 50


def convert_path_to_peptide_borders(
    pred: List[int],
    start_state: int,
    stop_state: int,
    offset: int = 0,
) -> List[Tuple[int, int]]:
    '''Given a Viterbi state sequence, return (start, end) tuples for each segment.'''
    seq_peptides = []
    is_peptide   = False
    peptide_start = 0

    for pos, p in enumerate(pred):
        if p == start_state and not is_peptide:
            is_peptide    = True
            peptide_start = pos
        elif p == stop_state and is_peptide:
            is_peptide = False
            seq_peptides.append((peptide_start + offset, pos + offset))

    if is_peptide:
        seq_peptides.append((peptide_start + offset, pos + offset))

    return seq_peptides


def parse_coordinate_string(coordinate_string: str) -> List[Tuple[int, int]]:
    if not coordinate_string:
        return []
    result = []
    for coords in coordinate_string.split(','):
        s, e = coords.split('-')
        result.append((int(s.lstrip('(')), int(e.rstrip(')'))))
    return result


def get_counts_for_protein(
    true_start_stop: List[Tuple[int, int]],
    pred_start_stop: List[Tuple[int, int]],
    tolerance: int = 3,
) -> Tuple[int, int, int]:
    if len(pred_start_stop) == 0:
        return 0, len(true_start_stop), 0
    if len(true_start_stop) == 0:
        return 0, 0, len(pred_start_stop)

    starts, stops = zip(*true_start_stop)
    true_df = pd.DataFrame({'start': starts, 'stop': stops})
    true_df = true_df.sort_values(['start', 'stop'], ascending=[True, False])
    true_df['group']   = (true_df['stop'].cummax().shift() < true_df['start']).cumsum()
    true_df['matched'] = False

    starts, stops = zip(*pred_start_stop)
    pred_df = pd.DataFrame({'start': starts, 'stop': stops})
    pred_df['matched'] = False

    for ti, trow in true_df.iterrows():
        for pi, prow in pred_df.iterrows():
            if (trow.start - tolerance <= prow.start <= trow.start + tolerance and
                    trow.stop  - tolerance <= prow.stop  <= trow.stop  + tolerance):
                true_df.loc[ti, 'matched'] = True
                pred_df.loc[pi, 'matched'] = True
                break

    true_matched = true_df.groupby('group')['matched'].any()
    tp = int(true_matched.sum())
    fn = int(len(true_matched) - tp)
    fp = int((~pred_df['matched']).sum())
    return tp, fn, fp


def compute_peptide_finding_metrics(
    true_start_stop: List[List[Tuple[int, int]]],
    pred_start_stop: List[List[Tuple[int, int]]],
    tolerance: int = 3,
) -> Tuple[float, float, float]:
    assert len(true_start_stop) == len(pred_start_stop)
    tp = fn = fp = 0
    for t, p in zip(true_start_stop, pred_start_stop):
        _tp, _fn, _fp = get_counts_for_protein(t, p, tolerance)
        tp += _tp; fn += _fn; fp += _fp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def score_one_model(pickle_path: str, true_df: pd.DataFrame) -> List[dict]:
    '''Load one test_outputs pickle and compute metrics at tolerances 0–3.'''
    with open(pickle_path, 'rb') as f:
        probs, preds, labels, names = pickle.load(f)

    propeptide_borders = [
        convert_path_to_peptide_borders(
            pred,
            start_state=PROPEPTIDE_START_STATE,
            stop_state=PROPEPTIDE_END_STATE,
            offset=1,
        )
        for pred in preds
    ]

    prediction_df = pd.DataFrame({'pred_propeptides': propeptide_borders}, index=names)
    df = prediction_df.join(true_df[['true_propeptides']])

    metrics = []
    for tolerance in [0, 1, 2, 3]:
        prec, rec, f1 = compute_peptide_finding_metrics(
            df['true_propeptides'].tolist(),
            df['pred_propeptides'].tolist(),
            tolerance=tolerance,
        )
        metrics.append({
            'tolerance':              tolerance,
            'precision propeptides':  prec,
            'recall propeptides':     rec,
            'f1 propeptides':         f1,
        })
    return metrics


def discover_pickle_files(out_dir: str) -> List[str]:
    '''Return all test_outputs_outer*_inner*.pickle files in out_dir, sorted.'''
    files = sorted(
        os.path.join(out_dir, f)
        for f in os.listdir(out_dir)
        if f.startswith('test_outputs_outer') and f.endswith('.pickle')
    )
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir',   required=True,
                        help='Directory written by train_loop_crf.py (contains *.pickle files)')
    parser.add_argument('--data_file', default='data/labeled_sequences.csv',
                        help='CSV with protein_id index and propeptide_coordinates column')
    parser.add_argument('--save_csv',  default=True, action=argparse.BooleanOptionalAction,
                        help='Write CSV result files alongside the pickle directory')
    args = parser.parse_args()

    # ---- Load ground truth ----
    df = pd.read_csv(args.data_file, index_col='protein_id')
    df = df.fillna('')
    df['true_propeptides'] = df['propeptide_coordinates'].apply(parse_coordinate_string)

    # ---- Discover model outputs ----
    pickle_files = discover_pickle_files(args.out_dir)
    if not pickle_files:
        raise FileNotFoundError(
            f'No test_outputs_outer*_inner*.pickle files found in {args.out_dir}.\n'
            'Run train_loop_crf.py first to generate them.'
        )
    print(f'Found {len(pickle_files)} model output files.')

    # ---- Score each model ----
    all_metrics = []
    for pkl in tqdm(pickle_files):
        model_name = os.path.basename(pkl).replace('.pickle', '')
        metrics = score_one_model(pkl, df)
        for m in metrics:
            m['model'] = model_name
        all_metrics.extend(metrics)

    metrics_df = pd.DataFrame(all_metrics).set_index(['tolerance', 'model'])

    # ---- Aggregate ----
    means = metrics_df.groupby(level='tolerance').mean()
    stds  = metrics_df.groupby(level='tolerance').std()

    print('\n=== Mean metrics across all models ===')
    for tol in [0, 1, 2, 3]:
        row = means.loc[tol]
        sd  = stds.loc[tol]
        print(
            f'  tolerance={tol}  '
            f'F1={row["f1 propeptides"]:.4f}±{sd["f1 propeptides"]:.4f}  '
            f'P={row["precision propeptides"]:.4f}  '
            f'R={row["recall propeptides"]:.4f}'
        )

    if args.save_csv:
        out_csv     = os.path.join(args.out_dir, 'metrics_per_model.csv')
        out_csv_agg = os.path.join(args.out_dir, 'metrics_aggregated.csv')
        metrics_df.to_csv(out_csv)
        means.to_csv(out_csv_agg)
        print(f'\nSaved:\n  {out_csv}\n  {out_csv_agg}')


if __name__ == '__main__':
    main()
