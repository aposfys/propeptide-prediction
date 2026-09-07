'''
Is "propeptide" one class, or several? And what happens to the label set if the
5-50 length window is widened?

This is the question behind "can I just use a bigger dataset and a wider CRF
grammar". The grammar works -- a 101-state CRF over 2..100 trains, decodes and
scores. What is not obvious is whether the extra annotations it lets in are the
same KIND of thing as the ones already there.

They are not, and the evidence is in UniProt's own keywords. Three processes are
pooled under the single PROPEP feature key:

  2-4 aa     CAAX processing. Prenylation on a C-terminal cysteine, then RCE1
             removes the "aaX" tripeptide, then ICMT methylates the cysteine.
             UniProt annotates the removed tripeptide as PROPEP. The recognition
             signal is the CaaX box, there is no dibasic site, and the length is
             always 3.
  5-50 aa    Proprotein convertase processing. Secretory precursors cut at
             mono/dibasic sites by furin and its relatives. This is the class
             the DeepPeptide benchmark was filtered down to.
  51-100 aa  Zymogen prodomains. Folded inhibitory domains on proteases, often
             autocatalytically removed, acting as intramolecular chaperones.

A CRF with one "propeptide" label and one shared emission has to learn all three
from one set of transition parameters. That is a harder and less coherent task
than the one the published numbers describe, and a change in F1 after widening
the window cannot be attributed to the model.

The second finding is about NEGATIVES, and it is the one that actually blocks a
naive rebuild: widening the window reclassifies a large share of the proteins
the current dataset calls negative. See the last section.

Usage
-----
    python -m src.utils.propeptide_class_audit --cache propep_rich.tsv

Needs a richer download than the other audits (keywords, families, fragment
status, organism), so it uses its own cache file.
'''
import argparse
import collections
import csv
import io
import os
import re
import urllib.request

import pandas as pd

STREAM = ('https://rest.uniprot.org/uniprotkb/stream'
          '?query=%28ft_propep%3A%2A%29%20AND%20%28reviewed%3Atrue%29'
          '&fields=accession%2Cprotein_name%2Cft_propep%2Clength%2Ckeyword'
          '%2Cprotein_families%2Corganism_name%2Cfragment&format=tsv')

# Split the Propeptide cell into one chunk per feature so each keeps its own
# /evidence block. A single regex over the whole cell would attribute the first
# feature's evidence to all of them.
CHUNK = re.compile(r'PROPEP\s+([?<>]?\d+|\?)\.\.([?<>]?\d+|\?)(.*?)(?=PROPEP\s|\Z)', re.S)
VIRAL = re.compile(r'virus|viral|phage|viroid', re.I)

BINS = [('2-4', lambda n: 2 <= n <= 4),
        ('5-50', lambda n: 5 <= n <= 50),
        ('51-100', lambda n: 51 <= n <= 100),
        ('>100', lambda n: n > 100)]

# UniProt keywords that mark a mechanism rather than a length.
MARKERS = [('Cleavage on pair of basic residues', 'convertase / furin processing'),
           ('Prenylation', 'CAAX prenylation'),
           ('Methylation', 'terminal methylation'),
           ('Zymogen', 'zymogen prodomain'),
           ('Protease', 'the protein is itself a protease')]


def load(cache_path):
    if cache_path and os.path.isfile(cache_path):
        print(f'Reading cached UniProt table from {cache_path}')
        raw = open(cache_path, encoding='utf-8').read()
    else:
        print('Fetching reviewed PROPEP annotations with keywords from UniProt...')
        with urllib.request.urlopen(STREAM, timeout=900) as response:
            raw = response.read().decode('utf-8')
        if cache_path:
            open(cache_path, 'w', encoding='utf-8').write(raw)

    features = []
    for row in csv.DictReader(io.StringIO(raw), delimiter='\t'):
        keywords = {k.strip() for k in (row.get('Keywords') or '').split(';') if k.strip()}
        chain_length = int(row['Length']) if (row.get('Length') or '').isdigit() else 0
        for match in CHUNK.finditer(row.get('Propeptide') or ''):
            start, end = match.group(1), match.group(2)
            if '?' in start or '?' in end:
                continue
            start, end = int(start.lstrip('<>')), int(end.lstrip('<>'))
            features.append({
                'accession': row['Entry'], 'length': end - start + 1,
                'start': start, 'end': end, 'chain_length': chain_length,
                'keywords': keywords,
                'family': (row.get('Protein families') or '').split(',')[0].strip(),
                'fragment': bool((row.get('Fragment') or '').strip()),
                'viral': bool(VIRAL.search(row.get('Organism') or '')),
            })
    return features


def table(title, rows, features):
    print()
    print(title)
    header = f'{"":38}' + ''.join(f'{name:>10}' for name, _ in BINS)
    print(header)
    print('-' * len(header))
    for label, value_of in rows:
        line = f'{label[:36]:38}'
        for _, rule in BINS:
            selected = [f for f in features if rule(f['length'])]
            line += f'{value_of(selected):9.1f}%'
        print(line)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cache', default='propep_rich.tsv')
    parser.add_argument('--data_file', default='data/labeled_sequences.csv')
    parser.add_argument('--partitioning_file', default='data/graphpart_assignments.csv')
    args = parser.parse_args()

    features = load(args.cache)
    # DeepPeptide's own exclusions, confirmed in the paper: UniProt 2022_01,
    # "excluding viral proteins and protein fragments".
    usable = [f for f in features if not f['fragment'] and not f['viral']]
    print(f'{len(features)} PROPEP features with a defined length; '
          f'{len(usable)} after dropping fragments and viral proteins')

    table('Mechanism markers by length bin (share of features in proteins carrying the keyword)',
          [(f'{kw}  [{desc}]',
            lambda sel, k=kw: 100 * sum(1 for f in sel if k in f['keywords']) / max(1, len(sel)))
           for kw, desc in MARKERS], usable)

    def position(f):
        if f['start'] <= 3:
            return 'N-terminal'
        if f['chain_length'] and f['end'] >= f['chain_length'] - 2:
            return 'C-terminal'
        return 'internal'

    table('Where the propeptide sits in the chain',
          [(p, lambda sel, p=p: 100 * sum(1 for f in sel if position(f) == p) / max(1, len(sel)))
           for p in ('N-terminal', 'C-terminal', 'internal')], usable)

    print()
    print('Read those two tables together. The 2-4 bin is prenylated, methylated '
          'and\nC-terminal, which is CAAX processing: RCE1 removes the aaX '
          'tripeptide after\nprenylation. The 51-100 bin is zymogen prodomains on '
          'proteases, and it is\ninternal. Neither is the dibasic convertase '
          'cleavage the 5-50 window selects.')

    print()
    print('=' * 74)
    print('What a 2..100 positive set would contain')
    print('=' * 74)
    widened = [f for f in usable if 2 <= f['length'] <= 100]

    def caax(f):
        return (f['length'] <= 4 and 'Prenylation' in f['keywords']
                and f['chain_length'] and f['end'] >= f['chain_length'] - 1)

    groups = collections.Counter()
    for f in widened:
        if caax(f):
            groups['CAAX / aaX removal (RCE1)'] += 1
        elif f['length'] >= 51 and ({'Zymogen', 'Protease'} & f['keywords']):
            groups['zymogen prodomain (51-100)'] += 1
        elif 'Cleavage on pair of basic residues' in f['keywords']:
            groups['convertase, dibasic keyword'] += 1
        elif f['length'] <= 4:
            groups['other short (2-4)'] += 1
        elif f['length'] >= 51:
            groups['other long (51-100)'] += 1
        else:
            groups['5-50, mechanism not keyworded'] += 1
    print(f'\n{len(widened)} features:')
    for name, n in groups.most_common():
        print(f'  {name:36} {n:6}  {100*n/len(widened):5.1f}%')
    outside = sum(1 for f in widened if not 5 <= f['length'] <= 50)
    print(f'\n  outside the current 5..50 window: {outside} ({100*outside/len(widened):.1f}%)')

    print()
    print('=' * 74)
    print('The blocking problem: widening reclassifies the NEGATIVES')
    print('=' * 74)
    if not os.path.isfile(args.data_file):
        print(f'  (skipped: no {args.data_file})')
        return
    frame = pd.read_csv(args.data_file).fillna('')
    partitioning = pd.read_csv(args.partitioning_file)
    effective = frame[frame['protein_id'].isin(set(partitioning['AC']))]
    negatives = set(effective[effective['propeptide_coordinates'].astype(str).str.strip()
                              == '']['protein_id'].astype(str))
    by_accession = collections.defaultdict(list)
    for f in features:
        by_accession[f['accession']].append(f['length'])

    counts = collections.Counter()
    for accession in negatives:
        lengths = by_accession.get(accession)
        if not lengths:
            counts['still no propeptide'] += 1
        elif any(5 <= n <= 50 for n in lengths):
            counts['now has one inside 5..50'] += 1
        elif any(2 <= n <= 100 for n in lengths):
            counts['now has one in 2..100 but not 5..50'] += 1
        else:
            counts['now has one, outside 2..100'] += 1
    print(f'\n{len(negatives)} proteins the benchmark labels NEGATIVE:')
    for name, n in counts.most_common():
        print(f'  {name:40} {n:5}  {100*n/len(negatives):5.1f}%')
    wrong_now = counts['now has one inside 5..50']
    wrong_wide = wrong_now + counts['now has one in 2..100 but not 5..50']
    print(f'\n  wrong under the current 5..50 grammar : {wrong_now:5}  '
          f'({100*wrong_now/len(negatives):.1f}%)')
    print(f'  wrong under a 2..100 grammar          : {wrong_wide:5}  '
          f'({100*wrong_wide/len(negatives):.1f}%)')
    print('\n  Widening does not only ADD positives from outside the file. It '
          'relabels a\n  third of the proteins the model is currently trained to '
          'call negative. A\n  rebuild has to redo the negative set, not just '
          'extend the positive one.')


if __name__ == '__main__':
    main()
