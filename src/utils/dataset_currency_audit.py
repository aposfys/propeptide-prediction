'''
Is the benchmark out of date, and would rebuilding it from current UniProt help?

The distributed data (`data/labeled_sequences.csv`) was built from UniProt
2022_02. This asks three separate questions against the live UniProt REST API,
because they have different answers and get confused with each other:

  1. CURRENCY -- are the labels we have still what UniProt says?
  2. GROWTH   -- how many propeptide proteins exist now that are not in the file?
  3. FIDELITY -- does the file label everything UniProt annotates for the
                 proteins it DOES contain?

(3) is the one that turns up something. The 5-50 length filter was applied to
ANNOTATIONS, not to proteins: a protein with one propeptide in range and one
outside is kept, with the out-of-range one silently absent from its labels. So
the benchmark contains real, UniProt-annotated propeptides that are not in the
ground truth.

Usage
-----
    python -m src.utils.dataset_currency_audit                    # fetch live
    python -m src.utils.dataset_currency_audit --cache propep.tsv # reuse

The reviewed PROPEP download is the same one propeptide_length_audit.py uses, so
point --cache at the same file and neither needs the network twice.
'''
import argparse
import collections
import csv
import io
import os
import re
import time
import urllib.parse
import urllib.request

import pandas as pd

from .crf_label_utils import parse_coordinate_string

# The release the benchmark was built from. The directory name upstream ships in
# its default paths is `uniprot_12052022_cv_5_50`, i.e. 12 May 2022.
BENCHMARK_DATE = '2022-05-12'

STREAM = ('https://rest.uniprot.org/uniprotkb/stream'
          '?query=%28ft_propep%3A%2A%29%20AND%20%28reviewed%3Atrue%29'
          '&fields=accession%2Cft_propep%2Clength&format=tsv')

# PROPEP 105..122, with UniProt's uncertain endpoints (`?`, `<1`, `>240`).
# A feature with an unknown endpoint has no defined length and is excluded.
FEATURE = re.compile(r'PROPEP\s+([?<>]?\d+|\?)\.\.([?<>]?\d+|\?)')

VIRAL = re.compile(r'virus|viral|phage|viroid', re.I)


def load_uniprot(cache_path):
    if cache_path and os.path.isfile(cache_path):
        print(f'Reading cached UniProt table from {cache_path}')
        raw = open(cache_path, encoding='utf-8').read()
    else:
        print('Fetching reviewed PROPEP annotations from UniProt...')
        with urllib.request.urlopen(STREAM, timeout=600) as response:
            raw = response.read().decode('utf-8')
        if cache_path:
            open(cache_path, 'w', encoding='utf-8').write(raw)

    spans_by_accession = {}
    for row in csv.DictReader(io.StringIO(raw), delimiter='\t'):
        spans = []
        for match in FEATURE.finditer(row.get('Propeptide') or ''):
            start, end = match.group(1), match.group(2)
            if '?' in start or '?' in end:
                continue
            spans.append((int(start.lstrip('<>')), int(end.lstrip('<>'))))
        if spans:
            spans_by_accession[row['Entry']] = spans
    return spans_by_accession


def fetch_metadata(accessions, batch=100):
    '''UniProt metadata for a list of accessions. The API caps OR terms at 100.'''
    frames = []
    for i in range(0, len(accessions), batch):
        query = ' OR '.join(f'accession:{a}' for a in accessions[i:i + batch])
        url = ('https://rest.uniprot.org/uniprotkb/stream?query='
               + urllib.parse.quote(query)
               + '&fields=accession,length,fragment,protein_existence,'
                 'date_created,organism_name&format=tsv')
        with urllib.request.urlopen(url, timeout=300) as response:
            frames.append(pd.read_csv(io.StringIO(response.read().decode()), sep='\t'))
        time.sleep(0.2)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def in_window(span, low=5, high=50):
    return low <= span[1] - span[0] + 1 <= high


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data_file', default='data/labeled_sequences.csv')
    parser.add_argument('--partitioning_file', default='data/graphpart_assignments.csv')
    parser.add_argument('--cache', default='',
                        help='Path to save/reuse the UniProt TSV.')
    parser.add_argument('--test_partition', type=int, default=4)
    parser.add_argument('--skip_metadata', action='store_true',
                        help='Skip the per-accession metadata fetch, which is the '
                             'slow part (~16 requests).')
    args = parser.parse_args()

    frame = pd.read_csv(args.data_file).fillna('')
    partitioning = pd.read_csv(args.partitioning_file)
    partitioned = set(partitioning['AC'])
    effective = frame[frame['protein_id'].isin(partitioned)]
    has_propeptide = effective['propeptide_coordinates'].astype(str).str.strip() != ''

    print()
    print('=' * 72)
    print('1. WHAT THE BENCHMARK ACTUALLY IS')
    print('=' * 72)
    print(f'  rows in {args.data_file:38} {len(frame):6}')
    print(f'  rows in {args.partitioning_file:38} {len(partitioned):6}')
    print(f'  {"dropped by GraphPart (never used)":46} {len(frame)-len(effective):6}')
    print(f'  {"effective dataset":46} {len(effective):6}')
    print(f'  {"  of which carry a propeptide":46} {int(has_propeptide.sum()):6}')
    print(f'  {"  of which do not (negatives)":46} {int((~has_propeptide).sum()):6}')
    organisms = frame['organism'].astype(str)
    print(f'  {"distinct organisms":46} {organisms.nunique():6}')
    print(f'  {"viral / phage proteins":46} {int(organisms.str.contains(VIRAL).sum()):6}'
          '   <- zero is a deliberate exclusion, not chance')

    current = load_uniprot(args.cache)
    print()
    print('=' * 72)
    print('2. CURRENCY -- are the labels we have still correct?')
    print('=' * 72)
    print('  Current spans are restricted to 5..50 for this comparison, so the')
    print("  length filter is not miscounted as an annotation change.")
    counts = collections.Counter()
    for _, row in effective[has_propeptide].iterrows():
        accession = str(row['protein_id'])
        old = {tuple(s) for s in parse_coordinate_string(
            str(row['propeptide_coordinates']), merge_overlaps=True)}
        if accession not in current:
            counts['no longer annotated'] += 1
            continue
        new = {s for s in current[accession] if in_window(s)}
        if old == new:
            counts['identical'] += 1
        elif old < new:
            counts['propeptides added'] += 1
        elif new < old:
            counts['propeptides removed'] += 1
        else:
            counts['coordinates changed'] += 1
    total = int(has_propeptide.sum())
    for label in ('identical', 'propeptides added', 'propeptides removed',
                  'coordinates changed', 'no longer annotated'):
        n = counts[label]
        print(f'  {label:28} {n:6}  {100*n/total:5.1f}%')
    moved = total - counts['identical']
    print(f'\n  labels that would move in a rebuild: {moved} ({100*moved/total:.1f}%)')

    print()
    print('=' * 72)
    print('3. GROWTH -- how much bigger would a rebuild be?')
    print('=' * 72)
    all_accessions = set(frame['protein_id'].astype(str))
    absent = sorted(set(current) - all_accessions)
    eligible = [a for a in absent if any(in_window(s) for s in current[a])]
    print(f'  reviewed proteins with PROPEP, today      {len(current):6}')
    print(f'  absent from the benchmark entirely        {len(absent):6}')
    print(f'    of those, >=1 propeptide inside 5..50   {len(eligible):6}'
          '   <- the only ones the current grammar could use')
    print()
    print('  A rebuild under different filters:')
    for label, rule in (
            ("DeepPeptide's rule (>=1 propeptide in 5..50)",
             lambda s: any(in_window(x) for x in s)),
            ('strict (every propeptide in 5..50)',
             lambda s: all(in_window(x) for x in s)),
            ('widened grammar (every propeptide in 2..105)',
             lambda s: all(in_window(x, 2, 105) for x in s)),
            ('no length filter at all', lambda s: True)):
        n = sum(1 for a in current if rule(current[a]))
        print(f'    {label:46} {n:6}  ({n/total:.1f}x)')

    if not args.skip_metadata and eligible:
        print()
        print(f'  Why are those {len(eligible)} eligible proteins not in the benchmark?')
        print('  (fetching metadata, ~16 requests)')
        metadata = fetch_metadata(eligible)
        organism = metadata['Organism'].astype(str)
        viral = organism.str.contains(VIRAL)
        fragment = metadata['Fragment'].fillna('').astype(str).str.len().gt(0)
        created = pd.to_datetime(metadata['Date of creation'], errors='coerce')
        recent = created > BENCHMARK_DATE
        for label, mask in (('viral / phage', viral), ('fragment', fragment),
                            (f'created after {BENCHMARK_DATE}', recent)):
            print(f'    {label:34} {int(mask.sum()):6}  {100*mask.mean():5.1f}%')
        explained = viral | fragment | recent
        print(f'    {"explained by those three":34} {int(explained.sum()):6}  '
              f'{100*explained.mean():5.1f}%')
        print(f'    {"UNEXPLAINED":34} {int((~explained).sum()):6}  '
              f'{100*(~explained).mean():5.1f}%')
        print(f'\n    -> {int((~explained).sum())} candidate additions, '
              f'+{100*int((~explained).sum())/total:.1f}% on the {total} the')
        print('       benchmark uses. Many are near-identical strain variants and')
        print('       homology-inferred annotations, so the usable gain after')
        print('       deduplication and GraphPart is smaller still.')

    print()
    print('=' * 72)
    print('4. FIDELITY -- does the file label everything UniProt annotates?')
    print('=' * 72)
    print('  The 5..50 filter was applied to ANNOTATIONS, not proteins. A protein')
    print('  with one propeptide in range and one outside is KEPT, with the')
    print('  out-of-range one absent from its labels.')
    proteins_affected = spans_missing = 0
    for _, row in effective[has_propeptide].iterrows():
        accession = str(row['protein_id'])
        if accession not in current:
            continue
        old = {tuple(s) for s in parse_coordinate_string(
            str(row['propeptide_coordinates']), merge_overlaps=True)}
        missing = [s for s in current[accession] if not in_window(s) and s not in old]
        if missing:
            proteins_affected += 1
            spans_missing += len(missing)
    print()
    print(f'  proteins carrying an unlabelled real propeptide  {proteins_affected:6}'
          f'  ({100*proteins_affected/total:.1f}%)')
    print(f'  unlabelled real propeptide spans                 {spans_missing:6}')

    merged = frame.merge(partitioning, left_on='protein_id', right_on='AC')
    test = merged[merged['cluster'] == args.test_partition]
    labelled = short = long_ = 0
    for _, row in test.iterrows():
        accession = str(row['protein_id'])
        old = {tuple(s) for s in parse_coordinate_string(
            str(row['propeptide_coordinates']), merge_overlaps=True)}
        labelled += len(old)
        for span in current.get(accession, []):
            length = span[1] - span[0] + 1
            if span in old or in_window(span):
                continue
            if length < 5:
                short += 1
            else:
                long_ += 1
    print()
    print(f'  In the test partition (cluster {args.test_partition}, {len(test)} proteins):')
    print(f'    labelled propeptide spans                      {labelled:6}')
    print(f'    unlabelled, shorter than 5 residues            {short:6}')
    print(f'    unlabelled, longer than 50 residues            {long_:6}')
    print()
    print('  Both groups are OUTSIDE the 5..50 the grammar can emit, and a hit')
    print('  needs both boundaries within tolerance -- so the model cannot score')
    print('  a true positive on any of them whatever it predicts. They therefore')
    print('  do NOT inflate the reported false-positive count directly. What they')
    print('  cost is EXTERNAL VALIDITY: the benchmark measures performance on a')
    print('  length-filtered slice of the propeptide population, not on it.')


if __name__ == '__main__':
    main()
