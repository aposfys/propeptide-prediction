'''
How big does the CRF grammar need to be?

The head is a state-space CRF whose states encode position within a propeptide,
so the number of states is a hard cap on the length of propeptide the model can
represent. The propeptide-only configuration uses 51 states: background 0 plus
states 1-50 for a propeptide of length 5..50 (`crf_label_utils.peptide_list_to_label_sequence`
with `min_len=5, max_len=50`).

That 5..50 window is inherited from Teufel et al., who state in the DeepPeptide
Methods that they "filtered the peptide annotations for a length range of 5-50
AAs and discarded all proteins that have no peptides within this range", and
report that this covers 90% of peptide but only **63%** of propeptide
annotations.

This script measures both distributions so the grammar can be sized against
evidence rather than inheritance:

  * the DISTRIBUTED BENCHMARK (`data/labeled_sequences.csv`), which is already
    filtered to 5..50 -- so measuring the cap against it is circular and will
    always say the cap is fine;
  * the UNFILTERED SwissProt PROPEP annotations pulled live from the UniProt
    REST API, which is the population the cap actually excludes.

The distinction matters because the two questions have opposite answers:
enlarging the grammar cannot improve a score on the distributed benchmark
(nothing there is out of range), and can only matter for a dataset rebuilt from
UniProt without the 5..50 filter.

Usage
-----
    python -m src.utils.propeptide_length_audit                    # fetch live
    python -m src.utils.propeptide_length_audit --cache propep.tsv # reuse a download
    python -m src.utils.propeptide_length_audit --benchmark_only   # no network

Note on reproducibility: UniProt is a moving target, so the absolute percentages
drift between releases. The benchmark was built from the 2022_02 release. The
run recorded in GRAMMAR.md used the 2026_04 reviewed set and reproduced Teufel
et al.'s 63% to within a point, which is the check that matters -- the shape of
the distribution is stable even though the counts are not.
'''
import argparse
import os
import re
import sys
import urllib.request
from collections import Counter

# Reviewed (SwissProt) entries carrying at least one PROPEP feature, with the
# feature ranges and the chain length. The stream endpoint paginates internally,
# so this is one request rather than a cursor loop.
UNIPROT_STREAM = (
    'https://rest.uniprot.org/uniprotkb/stream'
    '?query=%28ft_propep%3A%2A%29%20AND%20%28reviewed%3Atrue%29'
    '&fields=accession%2Cft_propep%2Clength'
    '&format=tsv'
)

# PROPEP 105..122 -- but UniProt also writes uncertain endpoints as `?`, `<1`
# and `>240`. A feature with an unknown endpoint has no defined length, so it is
# counted separately rather than guessed at.
FEATURE = re.compile(r'PROPEP\s+([?<>]?\d+|\?)\.\.([?<>]?\d+|\?)')


def _parse_feature_column(text):
    '''Yield propeptide lengths from one UniProt `Propeptide` cell.

    Returns (lengths, n_unknown). An endpoint written with `?` is unknown, not
    zero, so it is excluded from the length distribution and reported instead.
    '''
    lengths, unknown = [], 0
    for match in FEATURE.finditer(text or ''):
        start, end = match.group(1), match.group(2)
        if '?' in start or '?' in end:
            unknown += 1
            continue
        lengths.append(int(end.lstrip('<>')) - int(start.lstrip('<>')) + 1)
    return lengths, unknown


def _percentile(sorted_values, q):
    '''Nearest-rank percentile. Avoids a numpy dependency for one call.'''
    if not sorted_values:
        return 0
    index = max(0, min(len(sorted_values) - 1,
                       int(round(q / 100.0 * (len(sorted_values) - 1)))))
    return sorted_values[index]


def load_uniprot(cache_path):
    '''Fetch (or reuse) the reviewed PROPEP table and return per-protein lengths.'''
    if cache_path and os.path.isfile(cache_path):
        print(f'Reading cached UniProt table from {cache_path}')
        raw = open(cache_path, encoding='utf-8').read()
    else:
        print('Fetching reviewed PROPEP annotations from UniProt '
              '(one request, usually 10-60 s)...')
        with urllib.request.urlopen(UNIPROT_STREAM, timeout=600) as response:
            raw = response.read().decode('utf-8')
        if cache_path:
            open(cache_path, 'w', encoding='utf-8').write(raw)
            print(f'  cached to {cache_path}')

    lines = raw.rstrip('\n').split('\n')
    header = lines[0].split('\t')
    try:
        feature_col = header.index('Propeptide')
    except ValueError:
        raise SystemExit(f'Unexpected UniProt header: {header}')

    per_protein, unknown_total = [], 0
    for line in lines[1:]:
        fields = line.split('\t')
        if len(fields) <= feature_col:
            continue
        lengths, unknown = _parse_feature_column(fields[feature_col])
        unknown_total += unknown
        if lengths:
            per_protein.append(lengths)

    return per_protein, unknown_total


def load_benchmark(csv_path):
    '''Per-protein propeptide lengths from the distributed benchmark CSV.'''
    import pandas as pd
    from src.utils.crf_label_utils import parse_coordinate_string

    frame = pd.read_csv(csv_path).fillna('')
    per_protein = []
    for coordinate_string in frame['propeptide_coordinates'].tolist():
        spans = parse_coordinate_string(coordinate_string, merge_overlaps=True)
        lengths = [stop - start + 1 for start, stop in spans]
        if lengths:
            per_protein.append(lengths)
    return per_protein


def report(per_protein, label, unknown=0):
    '''Print the coverage table that decides the grammar size.'''
    flat = sorted(length for lengths in per_protein for length in lengths)
    n_features, n_proteins = len(flat), len(per_protein)

    print()
    print('=' * 72)
    print(f'{label}: {n_features} propeptide features across {n_proteins} proteins')
    if unknown:
        print(f'  ({unknown} further features have an uncertain endpoint and no '
              f'defined length)')
    print('=' * 72)
    if not flat:
        return

    print(f'  median {_percentile(flat, 50)}   '
          f'mean {sum(flat)/n_features:.1f}   max {flat[-1]}')
    print()
    print('  Where the mass sits:')
    below = sum(1 for x in flat if x < 5)
    above = sum(1 for x in flat if x > 50)
    print(f'    shorter than min_len 5 : {below:6d}  {100*below/n_features:6.2f}%')
    print(f'    inside 5..50           : {n_features-below-above:6d}  '
          f'{100*(n_features-below-above)/n_features:6.2f}%')
    print(f'    longer than max_len 50 : {above:6d}  {100*above/n_features:6.2f}%')

    print()
    print('  Coverage by candidate grammar. `states` is max_len + 1 (background).')
    print('  Viterbi is O(L * states^2), so the cost column is the decode-time')
    print('  multiplier against the current 51-state grammar.')
    print()
    print(f'    {"grammar":<14} {"states":>6} {"cost":>6} {"features":>9} {"proteins":>9}')
    print(f'    {"-"*14} {"-"*6:>6} {"-"*6:>6} {"-"*9:>9} {"-"*9:>9}')
    candidates = [(5, 50), (5, 60), (5, 105), (5, 262),
                  (2, 50), (2, 60), (2, 105), (2, 150), (2, 262),
                  (1, 50), (1, 105), (1, 262)]
    for min_len, max_len in candidates:
        states = max_len + 1
        feature_cov = 100 * sum(1 for x in flat if min_len <= x <= max_len) / n_features
        protein_cov = 100 * sum(
            1 for lengths in per_protein
            if all(min_len <= x <= max_len for x in lengths)) / n_proteins
        marker = '  <- current' if (min_len, max_len) == (5, 50) else ''
        print(f'    {min_len:>3}..{max_len:<9} {states:>6} '
              f'{(states/51.0)**2:>5.1f}x {feature_cov:>8.2f}% '
              f'{protein_cov:>8.2f}%{marker}')

    print()
    print('  Length needed to cover a target share of all annotations:')
    for target in (63, 80, 90, 95, 99):
        print(f'    {target:>2}% of features -> max_len {_percentile(flat, target):>4} '
              f'({_percentile(flat, target)+1} states)')

    print()
    print('  Short tail (the part min_len excludes, not max_len):')
    counts = Counter(x for x in flat if x < 5)
    for length in sorted(counts):
        print(f'    length {length}: {counts[length]:6d}  {100*counts[length]/n_features:5.2f}%')


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data_file', default='data/labeled_sequences.csv',
                        help='The distributed benchmark CSV.')
    parser.add_argument('--cache', default='',
                        help='Path to save/reuse the UniProt TSV, so a rerun '
                             'needs no network.')
    parser.add_argument('--benchmark_only', action='store_true',
                        help='Skip the UniProt request. Note that the benchmark '
                             'is already filtered to 5..50, so this alone cannot '
                             'answer the sizing question.')
    args = parser.parse_args()

    if os.path.isfile(args.data_file):
        report(load_benchmark(args.data_file),
               f'DISTRIBUTED BENCHMARK ({args.data_file}) -- pre-filtered to 5..50')
    else:
        print(f'No benchmark CSV at {args.data_file}, skipping that half.',
              file=sys.stderr)

    if args.benchmark_only:
        print('\n--benchmark_only: the UniProt half was skipped. The benchmark '
              'is already filtered to the very window under test, so its '
              'coverage figures are circular by construction.')
        return

    per_protein, unknown = load_uniprot(args.cache)
    report(per_protein, 'UNFILTERED SwissProt PROPEP (UniProt REST, live)', unknown)

    print()
    print('Read the two tables against each other. The benchmark says the '
          'current grammar is\nample; UniProt says it excludes about a third of '
          'real propeptide annotations. Both\nare true, and only the second one '
          'is about the grammar -- see GRAMMAR.md.')


if __name__ == '__main__':
    main()
