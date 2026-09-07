'''
Validate a built dataset against UniProt, and report what it covers.

`build_dataset.py` makes several decisions -- a length window, dropped CAAX
spans, rejected proteins, excluded viruses and fragments. Every one of them is a
claim about the data, and a claim that has not been checked is a guess. This
checks them.

Seven checks, each of which can fail:

  1  LABELS ARE COMPLETE. For every protein kept, the propeptide coordinates in
     the file equal UniProt's non-CAAX PROPEP spans exactly. This is the check
     the distributed benchmark fails: it keeps 548 proteins carrying 715 real
     propeptides its labels omit, because the 5-50 filter was applied per
     ANNOTATION rather than per protein.
  2  EVERY SPAN IS IN RANGE, so the grammar can represent all of them.
  3  SEQUENCES MATCH UniProt, so the labels index the right residues.
  4  NO FRAGMENTS.
  5  NO VIRUSES.
  6  NEGATIVES HAVE NO PROPEPTIDE.
  7  DUPLICATE SEQUENCES are reported. Identical sequences under different
     accessions become one embedding file (they are keyed by sequence hash) and
     can straddle a GraphPart partition boundary.

Then it reports COVERAGE: what share of the eligible UniProt population the file
actually contains, which is the number that justifies the window.

Usage
-----
    python -m src.utils.validate_dataset --data_dir data_v2 --cache_dir .
    python -m src.utils.validate_dataset --data_dir data --max_len 50   # the old one
'''
import argparse
import collections
import csv
import io
import os
import re
import sys
import urllib.parse
import urllib.request

import pandas as pd

FEATURE = re.compile(r'PROPEP\s+([?<>]?\d+|\?)\.\.([?<>]?\d+|\?)')
VIRAL = re.compile(r'virus|viral|phage|viroid', re.I)
FIELDS = 'accession,sequence,length,organism_name,fragment,keyword,ft_propep'
STREAM = 'https://rest.uniprot.org/uniprotkb/stream?query={q}&fields={f}&format=tsv'

failures = []


def check(condition, message, detail=''):
    print(('  PASS  ' if condition else '  FAIL  ') + message)
    if detail and not condition:
        print(f'        {detail}')
    if not condition:
        failures.append(message)


def load_uniprot(cache_path):
    if cache_path and os.path.isfile(cache_path):
        raw = open(cache_path, encoding='utf-8').read()
    else:
        url = STREAM.format(q=urllib.parse.quote('(ft_propep:*) AND (reviewed:true)'),
                            f=urllib.parse.quote(FIELDS))
        print('Fetching UniProt...')
        with urllib.request.urlopen(url, timeout=1800) as response:
            raw = response.read().decode('utf-8')
        if cache_path:
            open(cache_path, 'w', encoding='utf-8').write(raw)

    out = {}
    for row in csv.DictReader(io.StringIO(raw), delimiter='\t'):
        keywords = {k.strip() for k in (row.get('Keywords') or '').split(';') if k.strip()}
        sequence = (row.get('Sequence') or '').strip()
        spans = []
        for match in FEATURE.finditer(row.get('Propeptide') or ''):
            start, end = match.group(1), match.group(2)
            if '?' in start or '?' in end:
                continue
            spans.append((int(start.lstrip('<>')), int(end.lstrip('<>'))))
        out[row['Entry']] = {
            'spans': spans, 'keywords': keywords, 'sequence': sequence,
            'fragment': bool((row.get('Fragment') or '').strip()),
            'viral': bool(VIRAL.search(row.get('Organism') or '')),
            'length': len(sequence),
        }
    return out


def is_caax(span, entry):
    length = span[1] - span[0] + 1
    return (length <= 4 and 'Prenylation' in entry['keywords']
            and entry['length'] and span[1] >= entry['length'] - 1)


def parse_coords(text):
    out = []
    for chunk in str(text).split(','):
        chunk = chunk.strip().strip('()')
        if not chunk or '-' not in chunk:
            continue
        a, b = chunk.split('-')
        out.append((int(a), int(b)))
    return out


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--min_len', type=int, default=5)
    parser.add_argument('--max_len', type=int, default=100)
    parser.add_argument('--cache_dir', default='')
    args = parser.parse_args()

    frame = pd.read_csv(os.path.join(args.data_dir, 'labeled_sequences.csv')).fillna('')
    uniprot = load_uniprot(os.path.join(args.cache_dir, 'validate_uniprot.tsv')
                           if args.cache_dir else '')
    print(f'\n{len(frame)} rows in {args.data_dir}; '
          f'{len(uniprot)} reviewed PROPEP entries in UniProt\n')

    positives = frame[frame['propeptide_coordinates'].astype(str).str.strip() != '']
    negatives = frame[frame['propeptide_coordinates'].astype(str).str.strip() == '']
    print(f'{len(positives)} positives, {len(negatives)} negatives\n')

    print('--- 1. labels are complete (no unlabelled real propeptide) ---')
    incomplete, extra, absent = [], [], 0
    for _, row in positives.iterrows():
        accession = str(row['protein_id'])
        entry = uniprot.get(accession)
        if entry is None:
            absent += 1
            continue
        expected = {s for s in entry['spans'] if not is_caax(s, entry)}
        got = set(parse_coords(row['propeptide_coordinates']))
        if expected - got:
            incomplete.append((accession, sorted(expected - got)))
        if got - expected:
            extra.append((accession, sorted(got - expected)))
    check(not incomplete,
          f'every kept protein carries ALL its non-CAAX UniProt propeptides '
          f'({len(incomplete)} violations)',
          '; '.join(f'{a}: missing {s}' for a, s in incomplete[:3]))
    check(not extra, f'no label is absent from UniProt ({len(extra)} violations)',
          '; '.join(f'{a}: extra {s}' for a, s in extra[:3]))
    if absent:
        print(f'        ({absent} accessions not in the current UniProt pull)')

    print('--- 2. every span is representable by the grammar ---')
    lengths = [b - a + 1 for _, row in positives.iterrows()
               for a, b in parse_coords(row['propeptide_coordinates'])]
    check(lengths and min(lengths) >= args.min_len and max(lengths) <= args.max_len,
          f'all {len(lengths)} spans inside {args.min_len}..{args.max_len} '
          f'(observed {min(lengths)}..{max(lengths)})')

    print('--- 3. sequences match UniProt ---')
    mismatched = sum(1 for _, row in frame.iterrows()
                     if str(row['protein_id']) in uniprot
                     and uniprot[str(row['protein_id'])]['sequence']
                     and uniprot[str(row['protein_id'])]['sequence'] != str(row['sequence']))
    check(mismatched == 0, f'sequences identical to UniProt ({mismatched} differ)')

    print('--- 4/5. fragments and viruses excluded ---')
    frag = sum(1 for a in frame['protein_id'].astype(str)
               if a in uniprot and uniprot[a]['fragment'])
    vir = sum(1 for a in frame['protein_id'].astype(str)
              if a in uniprot and uniprot[a]['viral'])
    check(frag == 0, f'no fragments ({frag} found)')
    check(vir == 0, f'no viral proteins ({vir} found)')

    print('--- 6. negatives really have no propeptide ---')
    bad = sum(1 for a in negatives['protein_id'].astype(str)
              if a in uniprot and uniprot[a]['spans'])
    check(bad == 0, f'no negative carries a UniProt propeptide ({bad} found)')

    print('--- 7. duplicate sequences ---')
    counts = collections.Counter(frame['sequence'].astype(str))
    duplicated = {s: n for s, n in counts.items() if n > 1}
    print(f'  {"warn" if duplicated else "ok  "}   '
          f'{len(duplicated)} sequences appear more than once '
          f'({sum(duplicated.values()) - len(duplicated)} redundant rows)')
    if duplicated:
        print('        they share one embedding file and can straddle a GraphPart '
              'boundary; GraphPart should collapse them, verify after partitioning')

    print('\n=== COVERAGE: what this file contains of the eligible population ===')
    eligible = {a: e for a, e in uniprot.items()
                if not e['fragment'] and not e['viral']
                and any(not is_caax(s, e) for s in e['spans'])}
    in_window = {a: e for a, e in eligible.items()
                 if all(args.min_len <= s[1] - s[0] + 1 <= args.max_len
                        for s in e['spans'] if not is_caax(s, e))}
    held = set(positives['protein_id'].astype(str))
    all_spans = [s for e in eligible.values() for s in e['spans'] if not is_caax(s, e)]
    # Two different quantities, and conflating them flatters a file that omits
    # labels. `present` is what UniProt says about the proteins the file holds;
    # `labelled` is what the file actually asserts. The distributed benchmark's
    # gap between them is 564 proteins' worth of real propeptides it does not
    # label, which is the defect this whole rebuild exists to remove.
    present_spans = [s for a in held if a in uniprot
                     for s in uniprot[a]['spans'] if not is_caax(s, uniprot[a])]
    labelled_spans = [s for _, row in positives.iterrows()
                      for s in parse_coords(row['propeptide_coordinates'])]
    print(f'  reviewed, non-viral, non-fragment, at least one non-CAAX propeptide')
    print(f'    proteins   {len(eligible):6}')
    print(f'    spans      {len(all_spans):6}')
    print(f'  of those, every span inside {args.min_len}..{args.max_len}')
    print(f'    proteins   {len(in_window):6}   {100*len(in_window)/len(eligible):5.1f}% of eligible')
    print(f'  actually in this file')
    print(f'    proteins   {len(held):6}   {100*len(held)/len(eligible):5.1f}% of eligible, '
          f'{100*len(held)/max(1,len(in_window)):5.1f}% of in-window')
    print(f'    spans present in those proteins  {len(present_spans):6}   '
          f'{100*len(present_spans)/len(all_spans):5.1f}% of eligible spans')
    print(f'    spans the file actually LABELS   {len(labelled_spans):6}   '
          f'{100*len(labelled_spans)/len(all_spans):5.1f}% of eligible spans')
    gap = len(present_spans) - len(labelled_spans)
    if gap:
        print(f'    -> {gap} real propeptides sit inside kept proteins and are NOT '
              f'labelled')
    else:
        print(f'    -> no gap: every propeptide in a kept protein is labelled')

    print()
    print(f'{len(failures)} failure(s)')
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
