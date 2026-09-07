'''
Rescore an existing run's predictions, broken down by propeptide mechanism.

This is the cheapest test of the central claim in PAPER.md -- that UniProt's
`PROPEP` key pools unrelated processes and a pooled F1 hides them. It needs NO
training and NO GPU. It reads a `test_outputs.pickle` that a finished run already
wrote, assigns each test protein a mechanism from UniProt keywords, and reports
the metric separately per class.

If convertase-processed and zymogen/protease proteins already score differently
under the published model, the heterogeneity claim has empirical support before a
single sequence is re-downloaded. If they score the same, the claim is weaker and
that is worth knowing early.

Caveat this cannot fix: on the 5..50 benchmark the zymogen class is thin -- about
97 spans in the test partition against 560 convertase -- so read the zymogen row
as indicative. The class is properly populated only in the widened dataset, where
51-100 residue prodomains are in range.

Usage
-----
    python -m src.utils.score_by_mechanism results/esm2_prop_final/test_outputs.pickle
    python -m src.utils.score_by_mechanism results/*/test_outputs.pickle --cache propep_rich.tsv

Several runs can be passed at once; each is reported separately and then
aggregated, which is what a replicate group needs.
'''
import argparse
import collections
import csv
import inspect
import io
import os
import pickle
import statistics
import urllib.request

import numpy as np
import pandas as pd

from .crf_label_utils import parse_coordinate_string
from .manuscript_metrics import compute_all_metrics

# `end_state` is an addition on the multimodal branches. This script is meant to
# be droppable onto a checkout that predates it -- the whole point is to rescore
# runs that already exist, wherever they live -- so detect the signature instead
# of requiring the newer file. Without it the grammar is assumed to end at state
# 50, which is correct for every published run.
_HAS_END_STATE = 'end_state' in inspect.signature(compute_all_metrics).parameters

STREAM = ('https://rest.uniprot.org/uniprotkb/stream'
          '?query=%28ft_propep%3A%2A%29%20AND%20%28reviewed%3Atrue%29'
          '&fields=accession%2Ckeyword&format=tsv')


# Shipped so this needs no network. Regenerate it with --refresh when UniProt
# has moved on. It is 252 KB and covers every reviewed accession carrying a
# PROPEP feature, which is a superset of anything in the benchmark.
MECHANISM_TABLE = 'data/propeptide_mechanism.tsv'


def load_mechanisms(table_path, cache_path, refresh):
    '''accession -> mechanism label.

    Prefers the shipped table. The GPU nodes this runs on are busy and often
    firewalled, and a 13k-row lookup has no business being a network call every
    time -- the earlier version pulled the whole keyword set from UniProt on
    every invocation, which is what made this unusable mid-job.
    '''
    if not refresh and os.path.isfile(table_path):
        out = {}
        with open(table_path) as handle:
            next(handle)
            for line in handle:
                accession, mechanism_label = line.rstrip('\n').split('\t')
                out[accession] = mechanism_label
        return out

    if cache_path and os.path.isfile(cache_path):
        raw = open(cache_path, encoding='utf-8').read()
    else:
        print('Fetching UniProt keywords (only needed with --refresh)...')
        with urllib.request.urlopen(STREAM, timeout=600) as response:
            raw = response.read().decode('utf-8')
        if cache_path:
            open(cache_path, 'w', encoding='utf-8').write(raw)

    out = {}
    for row in csv.DictReader(io.StringIO(raw), delimiter='\t'):
        keywords = {k.strip() for k in (row.get('Keywords') or '').split(';') if k.strip()}
        if 'Cleavage on pair of basic residues' in keywords:
            out[row['Entry']] = 'convertase'
        elif 'Zymogen' in keywords or 'Protease' in keywords:
            out[row['Entry']] = 'zymogen_protease'
        else:
            out[row['Entry']] = 'unassigned'
    if refresh:
        with open(table_path, 'w') as handle:
            handle.write('accession\tmechanism\n')
            for accession, mechanism_label in sorted(out.items()):
                handle.write(f'{accession}\t{mechanism_label}\n')
        print(f'refreshed {table_path} ({len(out)} accessions)')
    return out


# Priority matters and is baked into the table: proteases that are themselves
# convertase-processed exist, and the dibasic keyword is the more specific
# statement about how THIS protein's propeptide is removed, so it wins.
# "unassigned" is honestly unassigned -- only about 28% of 5-50 features carry
# the dibasic keyword -- and must not be read as a fourth mechanism.
def mechanism(accession, table):
    return table.get(accession, 'not in current UniProt')


def score_one(path, frame, table, tolerances, end_state):
    '''Per-mechanism metrics for one test_outputs.pickle.'''
    probs, preds, labels, names = pickle.load(open(path, 'rb'))
    names = list(names)
    groups = collections.defaultdict(list)
    for i, name in enumerate(names):
        groups[mechanism(str(name), table)].append(i)

    out = {}
    for label, index in sorted(groups.items()):
        # Slice every parallel structure by the same index list, and rebuild the
        # truth frame for just this subset -- compute_all_metrics joins on names,
        # so passing the full frame would silently score the whole test set.
        subset_names = np.asarray([names[i] for i in index])
        subset = frame.loc[subset_names]
        extra = {'end_state': end_state} if _HAS_END_STATE else {}
        per_window = compute_all_metrics(
            [probs[i] for i in index] if isinstance(probs, list) else probs[index],
            [preds[i] for i in index],
            [labels[i] for i in index] if isinstance(labels, list) else labels[index],
            subset_names, subset, windows=tolerances, **extra)
        n_spans = int(sum(len(x) for x in subset['true_propeptides']))
        out[label] = {'n_proteins': len(index), 'n_spans': n_spans,
                      'metrics': dict(zip(tolerances, per_window))}
    return out


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('pickles', nargs='+', help='One or more test_outputs.pickle.')
    parser.add_argument('--data_file', default='data/labeled_sequences.csv')
    parser.add_argument('--mechanism_table', default=MECHANISM_TABLE,
                        help='Shipped accession -> mechanism table. No network needed.')
    parser.add_argument('--refresh', action='store_true',
                        help='Rebuild the table from UniProt. Only this needs a network.')
    parser.add_argument('--cache', default='propep_rich.tsv',
                        help='Optional raw UniProt TSV, used only with --refresh.')
    parser.add_argument('--tolerances', default='1,3')
    parser.add_argument('--end_state', type=int, default=50,
                        help="Last propeptide state of the grammar the run used, "
                             "i.e. its --max_peptide_len. Scoring a 101-state run "
                             "with 50 returns TRUNCATED spans, not empty ones.")
    args = parser.parse_args()
    tolerances = [int(x) for x in args.tolerances.split(',')]
    if not _HAS_END_STATE and args.end_state != 50:
        raise SystemExit(
            f'This checkout\'s compute_all_metrics has no end_state argument, so '
            f'it can only score a 50-state grammar, but --end_state {args.end_state} '
            'was given. Scoring a wider grammar here would silently truncate every '
            'span. Update src/utils/manuscript_metrics.py from a multimodal branch '
            'first.')

    frame = pd.read_csv(args.data_file, index_col='protein_id').fillna('')
    frame['true_propeptides'] = [parse_coordinate_string(x, merge_overlaps=True)
                                 for x in frame['propeptide_coordinates'].tolist()]
    frame['true_peptides'] = [[] for _ in range(len(frame))]
    table = load_mechanisms(args.mechanism_table, args.cache, args.refresh)

    per_run = {}
    for path in args.pickles:
        per_run[path] = score_one(path, frame, table, tolerances, args.end_state)
        print(f'\n=== {path} ===')
        for label, entry in per_run[path].items():
            row = ' '.join(
                f'F1@{t}={entry["metrics"][t]["f1 propeptides"]:.4f}' for t in tolerances)
            print(f'  {label:24} n={entry["n_proteins"]:5} spans={entry["n_spans"]:5}  {row}')

    if len(per_run) > 1:
        print(f'\n=== aggregate over {len(per_run)} runs (mean, sd) ===')
        labels = sorted({k for r in per_run.values() for k in r})
        for label in labels:
            for t in tolerances:
                values = [r[label]['metrics'][t]['f1 propeptides']
                          for r in per_run.values() if label in r]
                if len(values) < 2:
                    continue
                print(f'  {label:24} F1@{t}  mean {statistics.mean(values):.4f}  '
                      f'sd {statistics.stdev(values):.4f}  n={len(values)}')

    print('\nRead the convertase and zymogen rows against each other. A large gap')
    print('supports the claim that the pooled F1 averages over unlike tasks. On the')
    print('5..50 benchmark the zymogen row is thin (~97 spans) and is indicative')
    print('only; the class is properly populated in the widened dataset.')


if __name__ == '__main__':
    main()
