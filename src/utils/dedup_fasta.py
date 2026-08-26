'''
Collapse a FASTA to one record per distinct sequence, for make_embeddings.py.

WHY THIS EXISTS

fair-esm's FastaBatchedDataset.from_file asserts that sequence labels are
unique, and it derives the label by splitting the header on whitespace and
keeping the first token. data/protein_sequences.fasta carries headers like

    >P01210|organism=Homo sapiens (Human)|motif_cluster=13

which truncate to `P01210|organism=Homo`. Across the file that turns 12,897
distinct headers into 12,539 labels, and the extractor dies on the assert
before it embeds anything.

WHY DEDUPLICATING IS SAFE, AND NOT A CHANGE TO THE PIPELINE

The 875 colliding labels are the same protein listed once per motif_cluster.
Checked over the whole file: the number of collisions where the sequences
actually differ is ZERO.

make_embeddings.py never uses the label. It unpacks it (`label, seq = item`),
relabels every record to the constant "seq" before tokenising, and names the
output file `md5(seq).pt`. It also skips any sequence whose file already exists.
So running it over the deduplicated FASTA writes exactly the same set of files,
with exactly the same contents, as running it over the original would if the
assert were removed -- it just stops re-tokenising the 6,522 repeats.

That equivalence is the point. make_embeddings.py is byte-identical to upstream
DeepPeptide's and must stay that way for the ESM-2 baseline to be comparable to
the published pipeline, so the fix belongs in the input, not in the extractor.

    python src/utils/dedup_fasta.py \
        data/protein_sequences.fasta data/protein_sequences.dedup.fasta

Refuses to write if two records share a label but carry different sequences,
since that would mean the assumption above has stopped holding.
'''
import argparse
import sys
from hashlib import md5


def read_fasta(path):
    '''Yield (header, sequence). Joins wrapped lines -- the file wraps at 60
    columns, so a line-per-record reader silently truncates every sequence.'''
    header, buf = None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip('\n')
            if line.startswith('>'):
                if header is not None:
                    yield header, ''.join(buf)
                header, buf = line[1:], []
            else:
                buf.append(line)
    if header is not None:
        yield header, ''.join(buf)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('fasta_in')
    p.add_argument('fasta_out')
    p.add_argument('--wrap', type=int, default=60,
                   help='Line width for the output (0 for one line per record).')
    args = p.parse_args()

    records = list(read_fasta(args.fasta_in))
    if not records:
        raise SystemExit(f'{args.fasta_in}: no FASTA records found.')

    by_seq = {}          # md5 -> (header, sequence), first occurrence wins
    label_seqs = {}      # fair-esm's label -> set of sequences carrying it
    for header, seq in records:
        by_seq.setdefault(md5(seq.encode()).digest().hex(), (header, seq))
        label_seqs.setdefault(header.split()[0] if header.split() else '',
                              set()).add(seq)

    # The safety property: a label may repeat only if every record carrying it
    # has the same sequence. If that ever stops holding, deduplicating would
    # silently drop a real, distinct protein.
    conflicts = {lab for lab, s in label_seqs.items() if len(s) > 1}
    if conflicts:
        print(f'REFUSING: {len(conflicts)} labels carry DIFFERENT sequences, '
              f'e.g. {sorted(conflicts)[:3]}', file=sys.stderr)
        print('Deduplicating would drop distinct proteins. Fix the headers instead.',
              file=sys.stderr)
        sys.exit(1)

    with open(args.fasta_out, 'w') as out:
        for header, seq in by_seq.values():
            out.write(f'>{header}\n')
            if args.wrap:
                for i in range(0, len(seq), args.wrap):
                    out.write(seq[i:i + args.wrap] + '\n')
            else:
                out.write(seq + '\n')

    # Post-write check on the labels fair-esm will actually derive, so the
    # extractor cannot hit the same assert on the file this just produced.
    out_labels = [h.split()[0] if h.split() else '' for h, _ in by_seq.values()]
    if len(set(out_labels)) != len(out_labels):
        print(f'REFUSING: {len(out_labels) - len(set(out_labels))} duplicate labels '
              f'remain after deduplication -- fair-esm would still assert.',
              file=sys.stderr)
        sys.exit(1)

    print(f'{len(records)} records -> {len(by_seq)} distinct sequences')
    print(f'  dropped {len(records) - len(by_seq)} exact repeats '
          f'({len(conflicts)} label conflicts, all resolved)')
    print(f'  fair-esm labels unique: yes ({len(set(out_labels))})')
    print(f'written to {args.fasta_out}')


if __name__ == '__main__':
    main()
