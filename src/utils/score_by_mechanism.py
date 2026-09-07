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
import io
import os
import pickle
import statistics
import urllib.request

import numpy as np
import pandas as pd

from .crf_label_utils import parse_coordinate_string
from .manuscript_metrics import compute_all_metrics

STREAM = ('https://rest.uniprot.org/uniprotkb/stream'
          '?query=%28ft_propep%3A%2A%29%20AND%20%28reviewed%3Atrue%29'
          '&fields=accession%2Ckeyword&format=tsv')


def load_keywords(cache_path):
    '''accession -> set of UniProt keywords.'''
    if cache_path and os.path.isfile(cache_path):
        raw = open(cache_path, encoding='utf-8').read()
    else:
        print('Fetching UniProt keywords...')
        with urllib.request.urlopen(STREAM, timeout=600) as response:
            raw = response.read().decode('utf-8')
        if cache_path:
            open(cache_path, 'w', encoding='utf-8').write(raw)
    out = {}
    for row in csv.DictReader(io.StringIO(raw), delimiter='\t'):
        out[row['Entry']] = {k.strip() for k in (row.get('Keywords') or '').split(';')
                             if k.strip()}
    return out


def mechanism(accession, keywords):
    '''Assign one mechanism label, in priority order.

    Priority matters: proteases that are themselves convertase-processed exist,
    and the dibasic keyword is the more specific statement about how THIS
    protein's propeptide is removed, so it wins. "unassigned" is honestly
    unassigned -- only about 28% of 5-50 features carry the dibasic keyword --
    and must not be read as a fourth mechanism.
    '''
    kw = keywords.get(accession)
    if kw is None:
        return 'not in current UniProt'
    if 'Cleavage on pair of basic residues' in kw:
        return 'convertase (dibasic)'
    if 'Zymogen' in kw or 'Protease' in kw:
        return 'zymogen / protease'
    return 'unassigned'


def score_one(path, frame, keywords, tolerances, end_state):
    '''Per-mechanism metrics for one test_outputs.pickle.'''
    probs, preds, labels, names = pickle.load(open(path, 'rb'))
    names = list(names)
    groups = collections.defaultdict(list)
    for i, name in enumerate(names):
        groups[mechanism(str(name), keywords)].append(i)

    out = {}
    for label, index in sorted(groups.items()):
        # Slice every parallel structure by the same index list, and rebuild the
        # truth frame for just this subset -- compute_all_metrics joins on names,
        # so passing the full frame would silently score the whole test set.
        subset_names = np.asarray([names[i] for i in index])
        subset = frame.loc[subset_names]
        per_window = compute_all_metrics(
            [probs[i] for i in index] if isinstance(probs, list) else probs[index],
            [preds[i] for i in index],
            [labels[i] for i in index] if isinstance(labels, list) else labels[index],
            subset_names, subset, windows=tolerances, end_state=end_state)
        n_spans = int(sum(len(x) for x in subset['true_propeptides']))
        out[label] = {'n_proteins': len(index), 'n_spans': n_spans,
                      'metrics': dict(zip(tolerances, per_window))}
    return out


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('pickles', nargs='+', help='One or more test_outputs.pickle.')
    parser.add_argument('--data_file', default='data/labeled_sequences.csv')
    parser.add_argument('--cache', default='propep_rich.tsv')
    parser.add_argument('--tolerances', default='1,3')
    parser.add_argument('--end_state', type=int, default=50,
                        help="Last propeptide state of the grammar the run used, "
                             "i.e. its --max_peptide_len. Scoring a 101-state run "
                             "with 50 returns TRUNCATED spans, not empty ones.")
    args = parser.parse_args()
    tolerances = [int(x) for x in args.tolerances.split(',')]

    frame = pd.read_csv(args.data_file, index_col='protein_id').fillna('')
    frame['true_propeptides'] = [parse_coordinate_string(x, merge_overlaps=True)
                                 for x in frame['propeptide_coordinates'].tolist()]
    frame['true_peptides'] = [[] for _ in range(len(frame))]
    keywords = load_keywords(args.cache)

    per_run = {}
    for path in args.pickles:
        per_run[path] = score_one(path, frame, keywords, tolerances, args.end_state)
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
