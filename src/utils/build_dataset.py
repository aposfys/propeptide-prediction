'''
Build the rebuilt propeptide benchmark: reviewed UniProt, 5..100, mechanism-labelled.

A 101-state grammar on 5..50 data leaves half its states unvisited, so the
grammar and the dataset have to move together. This produces the dataset half.

WHAT IT KEEPS, AND WHY EACH FILTER IS THERE
    reviewed only            SwissProt. TrEMBL propeptide annotations are
                             overwhelmingly automatic.
    no fragments             a cleavage site cannot be localised on an
                             incomplete chain. Teufel et al. exclude these too.
    no viruses               viral polyproteins are cut by viral proteases with
                             unrelated specificity and would dominate the length
                             distribution. Teufel et al. exclude these too; the
                             distributed file has zero viral proteins across
                             1,660 organisms, though the repository never says so.
    span length 5..100       the CEILING is raised from 50 because 27.9% of
                             convertase-processed propeptides are longer than
                             50 -- the old cap truncates the target class rather
                             than selecting it. The FLOOR stays at 5 because
                             every CAAX feature is shorter than that, and CAAX
                             is a different reaction: prenylation, then RCE1
                             removing the aaX tripeptide, then methylation.
                             See GRAMMAR.md and DATASET_V2.md.

CAAX SPANS ARE DROPPED, NOT THEIR PROTEINS. A protein with a CAAX tripeptide and
a real propeptide keeps the propeptide and loses the tripeptide from its labels.
Dropping the whole protein would discard usable positives for the sake of an
annotation that belongs to another process.

MECHANISM LABELS are written to a sidecar, not into the label sequence. The CRF
predicts propeptide positions; the mechanism is for per-class reporting, and for
a later model that predicts segment and type jointly the way SignalP 6.0 does.

NEGATIVES are proteins with a mature-peptide annotation and no propeptide --
upstream's convention, and the sharpest available: they are known-cleaved
precursors, not arbitrary proteins. They are NOT reliable negatives and the
write-up must say so; this is positive-unlabeled data, and 37% of the 2022
negatives had gained a propeptide annotation by 2026.

AFTER THIS, RUN GRAPHPART. The output has no partitions. Numbers from an
unpartitioned split are not comparable to anything and will be optimistic.

Usage
-----
    python -m src.utils.build_dataset --out_dir data_v2
    python -m src.utils.build_dataset --out_dir /tmp/probe --limit 300
'''
import argparse
import collections
import csv
import io
import json
import os
import re
import sys
import urllib.parse
import urllib.request

FIELDS = ('accession,protein_name,sequence,length,organism_name,fragment,'
          'keyword,ft_propep,ft_peptide')
STREAM = 'https://rest.uniprot.org/uniprotkb/stream?query={q}&fields={f}&format=tsv'

POSITIVE_QUERY = '(ft_propep:*) AND (reviewed:true)'
NEGATIVE_QUERY = '(ft_peptide:*) AND (reviewed:true) NOT (ft_propep:*)'

FEATURE = re.compile(r'(PROPEP|PEPTIDE)\s+([?<>]?\d+|\?)\.\.([?<>]?\d+|\?)')
VIRAL = re.compile(r'virus|viral|phage|viroid', re.I)


def fetch(query, cache_path):
    if cache_path and os.path.isfile(cache_path):
        print(f'  reading {cache_path}')
        return open(cache_path, encoding='utf-8').read()
    url = STREAM.format(q=urllib.parse.quote(query), f=urllib.parse.quote(FIELDS))
    print(f'  fetching: {query}')
    with urllib.request.urlopen(url, timeout=1800) as response:
        raw = response.read().decode('utf-8')
    if cache_path:
        open(cache_path, 'w', encoding='utf-8').write(raw)
    return raw


def spans_of(cell, kind):
    '''(start, end) pairs of one feature kind. Uncertain endpoints are skipped:
    a feature with an unknown boundary has no defined span and guessing one
    would put a fabricated label into the ground truth.'''
    out = []
    for match in FEATURE.finditer(cell or ''):
        if match.group(1) != kind:
            continue
        start, end = match.group(2), match.group(3)
        if '?' in start or '?' in end:
            continue
        out.append((int(start.lstrip('<>')), int(end.lstrip('<>'))))
    return out


def is_caax(span, keywords, chain_length):
    length = span[1] - span[0] + 1
    return (length <= 4 and 'Prenylation' in keywords
            and chain_length and span[1] >= chain_length - 1)


def mechanism_of(keywords):
    if 'Cleavage on pair of basic residues' in keywords:
        return 'convertase'
    if 'Zymogen' in keywords or 'Protease' in keywords:
        return 'zymogen_protease'
    return 'unassigned'


def binary_string(spans, length):
    label = ['0'] * length
    for start, end in spans:
        for i in range(start - 1, min(end, length)):
            label[i] = '1'
    return ''.join(label)


def coordinate_string(spans):
    return ','.join(f'({a}-{b})' for a, b in sorted(spans))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--min_len', type=int, default=5)
    parser.add_argument('--max_len', type=int, default=100)
    parser.add_argument('--cache_dir', default='',
                        help='Reuse downloaded UniProt TSVs from here.')
    parser.add_argument('--limit', type=int, default=0,
                        help='Keep only the first N of each class, for a smoke test.')
    parser.add_argument('--no_negatives', action='store_true')
    parser.add_argument('--max_negatives', type=int, default=0,
                        help='Cap the negative set, sampled with --seed. The full '
                             'pool is 4,781 against 8,800 positives (35%% negative), '
                             'where the distributed benchmark was 1,231 against '
                             '6,392 (16%%). That shift moves precision and recall on '
                             'its own, so if you want the old balance set this to '
                             'about 1700. Using the whole pool is the default '
                             'because subsampling is an arbitrary choice that has '
                             'to be justified, and reporting it is easier than '
                             'defending a number.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for --max_negatives sampling.')
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    def cache(name):
        return os.path.join(args.cache_dir, name) if args.cache_dir else ''

    print('Positives:')
    positive_raw = fetch(POSITIVE_QUERY, cache('build_positives.tsv'))
    print('Negatives:')
    negative_raw = ('' if args.no_negatives
                    else fetch(NEGATIVE_QUERY, cache('build_negatives.tsv')))

    rows, mechanisms = [], {}
    stats = collections.Counter()

    def consider(record, want_propeptides):
        sequence = (record.get('Sequence') or '').strip()
        chain_length = len(sequence)
        if not sequence:
            stats['no_sequence'] += 1
            return
        if (record.get('Fragment') or '').strip():
            stats['fragment'] += 1
            return
        if VIRAL.search(record.get('Organism') or ''):
            stats['viral'] += 1
            return
        keywords = {k.strip() for k in (record.get('Keywords') or '').split(';') if k.strip()}

        propeptides = spans_of(record.get('Propeptide'), 'PROPEP')
        dropped_caax = [s for s in propeptides if is_caax(s, keywords, chain_length)]
        propeptides = [s for s in propeptides if s not in dropped_caax]
        stats['caax_spans_dropped'] += len(dropped_caax)

        if want_propeptides:
            if not propeptides:
                stats['no_usable_propeptide'] += 1
                return
            out_of_range = [s for s in propeptides
                            if not args.min_len <= s[1] - s[0] + 1 <= args.max_len]
            if out_of_range:
                # Reject the PROTEIN, not just the span. Keeping it would put an
                # unlabelled real propeptide inside the data, which is the defect
                # DATASET.md measures in the distributed benchmark: 548 proteins
                # carrying 715 propeptides their labels omit.
                stats['span_out_of_range'] += 1
                return
        elif propeptides:
            stats['negative_has_propeptide'] += 1
            return

        peptides = [s for s in spans_of(record.get('Peptide'), 'PEPTIDE')
                    if 1 <= s[0] <= s[1] <= chain_length]
        propeptides = [s for s in propeptides if 1 <= s[0] <= s[1] <= chain_length]

        rows.append({
            '': len(rows),
            'protein_name': (record.get('Protein names') or '')[:120],
            'sequence': sequence,
            'organism': record.get('Organism') or '',
            'is_peptide': binary_string(peptides, chain_length),
            'coordinates': coordinate_string(peptides),
            'is_propeptide': binary_string(propeptides, chain_length),
            'propeptide_coordinates': coordinate_string(propeptides),
            'protein_id': record['Entry'],
        })
        mechanisms[record['Entry']] = mechanism_of(keywords) if want_propeptides else 'negative'
        stats['positives' if want_propeptides else 'negatives'] += 1

    for raw, want in ((positive_raw, True), (negative_raw, False)):
        if not raw:
            continue
        kept = 0
        for record in csv.DictReader(io.StringIO(raw), delimiter='\t'):
            before = len(rows)
            consider(record, want)
            kept += len(rows) - before
            if args.limit and kept >= args.limit:
                break

    if not rows:
        raise SystemExit('Nothing passed the filters. Check the queries and cache.')

    if args.max_negatives:
        import random
        rng = random.Random(args.seed)
        positives = [r for r in rows if mechanisms[r['protein_id']] != 'negative']
        negatives = [r for r in rows if mechanisms[r['protein_id']] == 'negative']
        if len(negatives) > args.max_negatives:
            negatives = rng.sample(negatives, args.max_negatives)
        rows = positives + negatives
        for index, row in enumerate(rows):
            row[''] = index
        mechanisms = {r['protein_id']: mechanisms[r['protein_id']] for r in rows}
        stats['negatives'] = len(negatives)
        print(f'  --max_negatives {args.max_negatives}: sampled with seed {args.seed}')

    csv_path = os.path.join(args.out_dir, 'labeled_sequences.csv')
    with open(csv_path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Header format copied from the distributed FASTA, which reads
    #   >P01211|organism=Other|motif_cluster=13
    # GraphPart balances partitions on a label taken from the header, and the
    # original balanced on motif_cluster -- a 50-class clustering of PEPTIDE
    # motifs. That label cannot be reproduced (the clustering is not in the
    # repository) and would be the wrong one anyway: this is a propeptide-only
    # dataset, and motif_cluster describes mature peptides.
    #
    # So the mechanism label is used instead. It is the relevant stratification
    # here, and balancing on it keeps convertase, zymogen and negative proteins
    # evenly represented across partitions rather than letting a whole class land
    # in the test fold.
    #
    # The SEPARATION is unaffected by this choice: GraphPart's threshold forbids
    # >30% identity across partitions whatever it balances on. Only which
    # sequence lands where, and which are removed, depends on the label.
    fasta_path = os.path.join(args.out_dir, 'protein_sequences.fasta')
    with open(fasta_path, 'w') as handle:
        for row in rows:
            organism = (row['organism'] or 'Other').replace('|', ' ')
            handle.write(f">{row['protein_id']}|organism={organism}"
                         f"|mechanism={mechanisms[row['protein_id']]}\n"
                         f"{row['sequence']}\n")

    with open(os.path.join(args.out_dir, 'propeptide_mechanism.tsv'), 'w') as handle:
        handle.write('accession\tmechanism\n')
        for accession, label in sorted(mechanisms.items()):
            handle.write(f'{accession}\t{label}\n')

    json.dump({'min_len': args.min_len, 'max_len': args.max_len,
               'max_negatives': args.max_negatives or None,
               'seed': args.seed if args.max_negatives else None,
               'states_required': args.max_len + 1,
               'filters': ['reviewed', 'no fragments', 'no viruses',
                           f'all propeptide spans {args.min_len}..{args.max_len}',
                           'CAAX spans dropped, their proteins kept'],
               'counts': dict(stats), 'n_rows': len(rows),
               'mechanism_counts': dict(collections.Counter(mechanisms.values()))},
              open(os.path.join(args.out_dir, 'build_manifest.json'), 'w'), indent=2)

    print(f'\n=== {len(rows)} proteins written to {args.out_dir} ===')
    for key, value in stats.most_common():
        print(f'  {key:26} {value:6}')
    print('\n  mechanisms:', dict(collections.Counter(mechanisms.values())))
    print(f'\n  train with --max_peptide_len {args.max_len} '
          f'({args.max_len + 1} states)')
    print('\nNEXT, AND NOT OPTIONAL — homology-partition it.')
    print('These separation settings are the published ones and must not change:')
    print('needle (Needleman-Wunsch), 30% identity, 5 partitions. Switching to')
    print('the mmseqs2 backend would compare a different set of pairs and remove')
    print('a different set of sequences.\n')
    print(f'  graphpart needle -ff {fasta_path} \\')
    print(f'      -th 0.3 -pa 5 -ln mechanism \\')
    print(f'      -on {os.path.join(args.out_dir, "graphpart_assignments.csv")}')
    print('\n  -ln mechanism balances the classes across partitions. The original')
    print('  balanced on motif_cluster, a clustering of PEPTIDE motifs that is not')
    print('  in the repository and describes the wrong thing for a propeptide-only')
    print('  dataset. The separation guarantee does not depend on the label.')
    print('\n  Numbers from an unpartitioned split are optimistic and comparable')
    print('  to nothing.')


if __name__ == '__main__':
    main()
