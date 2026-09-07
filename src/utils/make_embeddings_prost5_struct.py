'''
ProstT5 embeddings from amino acids AND structure, rather than amino acids alone.

`make_embeddings.py` on this branch passes `<AA2fold>` + the residue sequence and
takes the encoder's last hidden state -- 1024 dims per residue, sequence-only.
ProstT5 is bilingual: it was trained on amino acids and on Foldseek 3Di, and it
accepts either language. The sequence-only arm therefore used half the model's
input vocabulary, and the ProstT5-vs-ESM-2 comparison in RESULTS.md never tested
whether structure helps -- the structural language was simply never spoken.

This script speaks both and concatenates:

    <AA2fold> + amino acids  -> encoder -> 1024 per residue   (the AA channel)
    <fold2AA> + 3Di          -> encoder -> 1024 per residue   (the 3Di channel)
    concatenate                          -> 2048 per residue

Pass `--embedding_dim 2048` to run.py. `--tracks aa` and `--tracks 3di` produce
the 1024-dim single-channel arms, so the three-way ablation is one flag.

WHY CONCATENATE RATHER THAN INTERLEAVE
Both languages tokenise one token per residue -- verified against the tokenizer:
`<fold2AA> d v v v v` yields ['<fold2AA>','d','v','v','v','v','</s>'], one token
each, no sentencepiece merging. So the two channels are position-aligned by
construction and can be concatenated per residue with no alignment step. That is
the property SaProt gets by building a joint vocabulary; here it falls out of the
tokeniser for free.

CASE IS LOAD-BEARING
ProstT5 distinguishes the two languages by CASE, not only by the prefix token.
Uppercase letters tokenise with the sentencepiece word-boundary marker (`▁D`) and
are read as amino acids; lowercase letters tokenise bare (`d`) and are read as
3Di. A 3Di string left uppercase is silently interpreted as an amino-acid
sequence -- same shapes, same runtime, wrong meaning, and a plausible F1 at the
end of it. `format_3di_for_prostt5` lowercases, and this is why.

MISSING STRUCTURE IS MASKED, NOT DROPPED
About 12% of the dataset has no usable AlphaFold model or fails the exact
sequence match. Those proteins keep their AA channel and get a ZERO 3Di channel.
Dropping them instead would change the GraphPart partitions and break
comparability with every sequence-only run -- the same policy as the ESM3
structure extractor, chosen so the two structure arms are answerable against each
other.

Zeroing is a real modelling compromise, not a no-op: a zero block is a value the
encoder never emits, so the head can in principle learn to detect "no structure"
and route around it. That is a confound the shuffled control below is there to
bound.

THE CONTROL
`--shuffle_3di` pairs each protein with a DIFFERENT protein's 3Di string, trimmed
or tiled to the right length. It keeps the input dimensionality, the zero-mask
pattern, the parameter count and the training budget identical, and destroys only
the correspondence between structure and sequence. Without it, a gain from 1024
to 2048 dims cannot be told apart from a gain from structural information: the
head got twice as many features either way. Run it. A real structural effect must
beat the shuffled control, not just the sequence-only arm.

Usage
-----
    python -m src.utils.fetch_afdb_structures --out_dir structures/
    python -m src.utils.make_3di --structures_dir structures/ --out_dir three_di/
    python -m src.utils.make_embeddings_prost5_struct \
        data/protein_sequences.fasta embeddings/prost5_aa3di \
        --three_di three_di/three_di.json
'''
import argparse
import json
import os
import pathlib
import random
import re
from hashlib import md5

import torch
from tqdm.auto import tqdm


def hash_aa_string(string: str) -> str:
    return md5(string.encode()).digest().hex()


def format_aa_for_prostt5(sequence: str) -> str:
    '''ProstT5's documented amino-acid input: rare residues to X, spaced, prefixed.

    Identical to make_embeddings.py's formatter on this branch, on purpose -- the
    AA channel here must be the same tensor the sequence-only arm was trained on,
    or the contrast measures the formatter instead of the structure.
    '''
    return '<AA2fold> ' + ' '.join(list(re.sub(r'[UZOB]', 'X', sequence.upper())))


def format_3di_for_prostt5(three_di: str) -> str:
    '''ProstT5's documented 3Di input: LOWERCASE, spaced, `<fold2AA>` prefix.

    The lowercasing is what tells the model this is structure and not sequence.
    See the module docstring.
    '''
    return '<fold2AA> ' + ' '.join(list(three_di.lower()))


def _read_fasta(path):
    sequences, current = {}, None
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line.startswith('>'):
                current = line[1:].split()[0]
                sequences[current] = ''
            elif current is not None:
                sequences[current] += line
    return sequences


def _encode_batch(model, tokenizer, device, formatted, lengths):
    '''One forward pass. Returns a list of (length, 1024) float32 CPU tensors.

    Offset 1 drops the language-prefix token, so position i of the output is
    residue i of the input. `.clone()` is not optional: without it torch.save
    serialises the whole padded batch storage behind each view, which is how the
    removed make_embeddings_prost5.py came to write ~400 MB per sequence.
    '''
    encoding = tokenizer.batch_encode_plus(
        list(formatted), add_special_tokens=True, padding='longest',
        return_tensors='pt').to(device)
    with torch.no_grad():
        output = model(encoding.input_ids, attention_mask=encoding.attention_mask)

    out = []
    for index, length in enumerate(lengths):
        vector = output.last_hidden_state[index, 1:length + 1].float().cpu().clone()
        # fp16 T5 can overflow to inf and then NaN, and both survive into the .pt
        # file looking like a normal embedding -- the run only shows it later as a
        # diverged loss. Zero them and say so.
        n_bad = int((~torch.isfinite(vector)).sum())
        if n_bad:
            print(f'WARNING: {n_bad} non-finite value(s) zeroed in a '
                  f'{length}-residue embedding. If this run used --half, redo it '
                  'in full precision.')
            vector[~torch.isfinite(vector)] = 0.0
        out.append(vector)
    return out


def generate(fasta_file, out_dir, three_di_path, tracks, shuffle_3di, model_name,
             half_precision, max_residues, max_seq_len, max_batch, seed):
    from transformers import T5EncoderModel, T5Tokenizer

    want_aa = tracks in ('aa', 'aa+3di')
    want_3di = tracks in ('3di', 'aa+3di')
    dimension = 1024 * (int(want_aa) + int(want_3di))

    three_di_by_hash = {}
    if want_3di:
        if not three_di_path:
            raise SystemExit('--three_di is required unless --tracks aa.')
        three_di_by_hash = json.load(open(three_di_path))
        n_real = sum(1 for v in three_di_by_hash.values() if v)
        print(f'3Di: {n_real}/{len(three_di_by_hash)} sequences have a real string; '
              f'the rest get a zero channel.')

    # Record what produced this directory. Which channels were fed is not
    # recoverable from the .pt files, and provenance must not depend on
    # remembering the command line.
    os.makedirs(out_dir, exist_ok=True)
    json.dump({'tracks': tracks, 'shuffle_3di': shuffle_3di, 'seed': seed,
               'three_di': three_di_path, 'model': model_name,
               'half_precision': half_precision, 'embedding_dim': dimension},
              open(os.path.join(out_dir, 'extraction_config.json'), 'w'), indent=2)

    sequences = _read_fasta(fasta_file)
    # Longest first, so an OOM shows up in the first minute rather than the last.
    ordered = sorted(sequences.items(), key=lambda kv: len(kv[1]), reverse=True)
    print(f'{len(ordered)} sequences, tracks={tracks}, {dimension} dims per residue')

    if shuffle_3di and want_3di:
        # Permute the mapping from sequence to 3Di, keeping the pool of real
        # strings and the set of hashes that have one. Derangement is not
        # enforced; with thousands of sequences the chance of a fixed point
        # mattering is negligible, and forcing one would itself be a bias.
        rng = random.Random(seed)
        keys = [k for k, v in three_di_by_hash.items() if v]
        values = [three_di_by_hash[k] for k in keys]
        rng.shuffle(values)
        three_di_by_hash = dict(three_di_by_hash)
        three_di_by_hash.update(dict(zip(keys, values)))
        print(f'--shuffle_3di: {len(keys)} 3Di strings permuted with seed {seed}. '
              'This is the CONTROL arm, not a result arm.')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Using device: {device}')
    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_name).to(device)
    if half_precision and device.type != 'cuda':
        print('WARNING: --half ignored on CPU; ProstT5 needs full precision there.')
        half_precision = False
    model = model.half() if half_precision else model.float()
    model.eval()

    def fit_to_length(string, length):
        '''Trim or tile a donor 3Di string to `length`. Shuffle control only.'''
        if len(string) >= length:
            return string[:length]
        return (string * (length // len(string) + 1))[:length]

    pending, n_saved, n_masked = [], 0, 0

    def flush():
        nonlocal pending, n_saved
        if not pending:
            return
        lengths = [len(record[1]) for record in pending]

        aa_vectors = _encode_batch(
            model, tokenizer, device,
            [format_aa_for_prostt5(record[1]) for record in pending],
            lengths) if want_aa else [None] * len(pending)

        # The 3Di channel is a SEPARATE forward pass, not a second half of the
        # same batch: the two languages have different prefixes and must not
        # share a padded tensor whose attention mask was built for the other.
        if want_3di:
            di_vectors = []
            real = [(i, r) for i, r in enumerate(pending) if r[2]]
            if real:
                di_out = _encode_batch(
                    model, tokenizer, device,
                    [format_3di_for_prostt5(r[2]) for _, r in real],
                    [len(r[2]) for _, r in real])
            else:
                di_out = []
            by_index = dict(zip([i for i, _ in real], di_out))
            for i, record in enumerate(pending):
                if i in by_index:
                    di_vectors.append(by_index[i])
                else:
                    di_vectors.append(torch.zeros(len(record[1]), 1024,
                                                  dtype=torch.float32))
        else:
            di_vectors = [None] * len(pending)

        for (digest, sequence, _), aa_vector, di_vector in zip(pending, aa_vectors,
                                                               di_vectors):
            parts = [part for part in (aa_vector, di_vector) if part is not None]
            embedding = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
            if embedding.shape != (len(sequence), dimension):
                raise RuntimeError(
                    f'refusing to write a {tuple(embedding.shape)} tensor for a '
                    f'{len(sequence)}-residue sequence; expected '
                    f'({len(sequence)}, {dimension})')
            torch.save(embedding, os.path.join(out_dir, f'{digest}.pt'))
            n_saved += 1
        pending = []

    for _, sequence in tqdm(ordered):
        digest = hash_aa_string(sequence)
        if os.path.isfile(os.path.join(out_dir, f'{digest}.pt')):
            n_saved += 1
            continue

        three_di = three_di_by_hash.get(digest) if want_3di else None
        if want_3di:
            if three_di and shuffle_3di:
                three_di = fit_to_length(three_di, len(sequence))
            if three_di and len(three_di) != len(sequence):
                # Only reachable in the shuffle path or from a stale three_di.json.
                # A misaligned structural channel is worse than none, so mask it.
                three_di = None
            if not three_di:
                n_masked += 1

        if len(sequence) > max_seq_len and pending:
            flush()
        pending.append((digest, sequence, three_di))
        if (len(pending) >= max_batch
                or sum(len(r[1]) for r in pending) >= max_residues
                or len(sequence) > max_seq_len):
            flush()

    # Flush outside the loop, not on a `last item` test inside it: on a resumed
    # run the final sequences are usually cached and `continue` skips the test,
    # which silently drops the trailing batch while still reporting success.
    flush()

    print(f'\nDone. {n_saved} embeddings in {out_dir} at {dimension} dims.')
    if want_3di:
        print(f'{n_masked} of them have a zero 3Di channel (no usable structure).')
    print(f'Pass --embedding_dim {dimension} to run.py.')


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('fasta_file', type=pathlib.Path)
    parser.add_argument('output_dir', type=pathlib.Path,
                        help='Fresh directory. Existing hashes are skipped, so '
                             'reusing a sequence-only directory writes nothing.')
    parser.add_argument('--three_di', default='',
                        help='three_di.json from make_3di.py.')
    parser.add_argument('--tracks', choices=['aa', '3di', 'aa+3di'], default='aa+3di',
                        help='aa = the 1024-dim sequence-only baseline; 3di = the '
                             '1024-dim structure-only arm; aa+3di = 2048-dim '
                             'concatenation (default).')
    parser.add_argument('--shuffle_3di', action='store_true',
                        help='CONTROL: pair each protein with another protein\'s '
                             '3Di string. Same dimensionality, same mask pattern, '
                             'no real structural correspondence. A structure '
                             'effect has to beat this, not just the AA arm.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for --shuffle_3di. Recorded in extraction_config.json.')
    parser.add_argument('--model', default='Rostlab/ProstT5')
    parser.add_argument('--half', action='store_true',
                        help='fp16, CUDA only. Check the non-finite warnings if used.')
    parser.add_argument('--max_seq_len', type=int, default=1000)
    parser.add_argument('--max_residues', type=int, default=4000)
    parser.add_argument('--max_batch', type=int, default=100)
    args = parser.parse_args()

    generate(args.fasta_file, args.output_dir, args.three_di, args.tracks,
             args.shuffle_3di, args.model, args.half, args.max_residues,
             args.max_seq_len, args.max_batch, args.seed)


if __name__ == '__main__':
    main()
