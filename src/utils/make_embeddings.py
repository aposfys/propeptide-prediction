'''
Generate ProstT5 per-residue embeddings and save as one file per sequence.
Keeps the MD5-hash filename convention the ESM-2 and ESM3 branches use, so the
dataset and training pipeline work unchanged — just point --embeddings_dir at
the ProstT5 output and pass --embedding_dim 1024.

This is the only ProstT5 extractor on the branch. An earlier duplicate,
make_embeddings_prost5.py, was removed: it was identical except for a missing
.clone() on the output slice, which made torch.save serialise the entire padded
batch storage behind the view (~400 MB per sequence instead of ~0.2 MB).

Model: Rostlab/ProstT5 (T5-encoder, 1024-dim per residue)
Ref: https://github.com/mheinzinger/ProstT5
'''
import os
import re
import argparse
import pathlib
from hashlib import md5

import torch
from tqdm.auto import tqdm


def hash_aa_string(string):
    return md5(string.encode()).digest().hex()


def format_for_prostt5(sequence: str) -> str:
    '''Apply ProstT5's documented input formatting to a raw AA sequence.

    Straight from the ProstT5 README: replace rare/ambiguous residues with X,
    put whitespace between all residues, and prepend <AA2fold> to declare the
    input is amino acids (rather than 3Di, which takes <fold2AA>).

    Single source of truth on purpose -- the batch extractor here and the
    single-sequence path in crf_models._esm_embed must format identically, or
    inference sees different tokens than training did.
    '''
    seq_clean = re.sub(r'[UZOB]', 'X', sequence.upper())
    return '<AA2fold> ' + ' '.join(list(seq_clean))


def _read_fasta(fasta_path):
    sequences = {}
    current_id = None
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                current_id = line[1:].split()[0]
                sequences[current_id] = ''
            elif current_id is not None:
                sequences[current_id] += line
    return sequences


def _flush_batch(batch, model, tokenizer, device, embeddings_dir, stats=None):
    '''Run one batch through the model and save per-residue embeddings.'''
    ids_batch, raw_seqs_batch, formatted_batch, lens_batch = zip(*batch)

    token_encoding = tokenizer.batch_encode_plus(
        list(formatted_batch),
        add_special_tokens=True,
        padding='longest',
        return_tensors='pt',
    ).to(device)

    try:
        with torch.no_grad():
            output = model(
                token_encoding.input_ids,
                attention_mask=token_encoding.attention_mask,
            )
    except RuntimeError as e:
        print(f'RuntimeError on batch starting with {ids_batch[0]} (len={lens_batch[0]}): {e}')
        return 0

    n_saved = 0
    for b_idx, (raw_seq_b, s_len) in enumerate(zip(raw_seqs_batch, lens_batch)):
        out_path = os.path.join(embeddings_dir, f'{hash_aa_string(raw_seq_b)}.pt')
        if os.path.isfile(out_path):
            n_saved += 1
            continue
        # Offset 1 skips the <AA2fold> prefix token; take exactly s_len residue positions
        emb = output.last_hidden_state[b_idx, 1:s_len + 1].float().cpu().clone()

        # Do not let a bad value pass silently. T5 in fp16 can overflow to inf
        # and then produce NaN, and both survive into the .pt file looking like
        # a normal embedding -- the training run only shows it much later as a
        # diverged loss. Zeroing NaN is what this script always did; the counter
        # and the warning are what make a broken --half run visible while it is
        # still cheap to redo in fp32.
        n_bad = int((~torch.isfinite(emb)).sum())
        if n_bad:
            print(f'WARNING: {ids_batch[b_idx]} (len={s_len}) has {n_bad} non-finite '
                  f'value(s) out of {emb.numel()} — zeroed. If this run used --half, '
                  'redo it in full precision.')
            if stats is not None:
                stats['bad_seqs'] += 1
                stats['bad_values'] += n_bad
        emb[~torch.isfinite(emb)] = 0.0

        torch.save(emb, out_path)
        n_saved += 1
    return n_saved


def generate_prost5_embeddings(
    fasta_file,
    embeddings_dir,
    model_name='Rostlab/ProstT5',
    half_precision=False,
    max_residues=4000,
    max_seq_len=1000,
    max_batch=100,
):
    from transformers import T5EncoderModel, T5Tokenizer

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Using device: {device}')

    print(f'Loading ProstT5 from {model_name}...')
    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_name).to(device)
    # Per the ProstT5 README: "only GPUs support half-precision currently; if you
    # want to run on CPU use full-precision". fp16 on CPU is not just slower --
    # several ops have no CPU kernel and raise at runtime. Downgrade rather than
    # let the run die an hour in.
    if half_precision and device.type != 'cuda':
        print('WARNING: --half ignored on CPU; ProstT5 requires full precision there.')
        half_precision = False
    # The README writes this as model.full(); no such method exists on nn.Module
    # or T5EncoderModel in current transformers, so .float() it is.
    model = model.half() if half_precision else model.float()
    print(f'Running in {"half (fp16)" if half_precision else "full (fp32)"} precision')
    model.eval()

    sequences = _read_fasta(fasta_file)
    # Sort longest-first so OOM failures surface early
    sorted_seqs = sorted(sequences.items(), key=lambda kv: len(kv[1]), reverse=True)
    n_total = len(sorted_seqs)
    print(f'Generating ProstT5 embeddings for {n_total} sequences')

    batch = []
    n_saved = 0
    stats = {'bad_seqs': 0, 'bad_values': 0}

    for seq_id, raw_seq in tqdm(sorted_seqs):
        seq_hash = hash_aa_string(raw_seq)
        if os.path.isfile(os.path.join(embeddings_dir, f'{seq_hash}.pt')):
            n_saved += 1
            continue

        seq_formatted = format_for_prostt5(raw_seq)
        seq_len = len(raw_seq)

        # If this sequence alone exceeds the residue budget, flush any accumulated
        # batch first so it gets processed in isolation.
        if seq_len > max_seq_len and batch:
            n_saved += _flush_batch(batch, model, tokenizer, device, embeddings_dir, stats)
            batch = []

        batch.append((seq_id, raw_seq, seq_formatted, seq_len))

        # n_res_batch counts what is already in the batch (current seq included).
        n_res_batch = sum(s_len for _, _, _, s_len in batch)
        flush = (
            len(batch) >= max_batch
            or n_res_batch >= max_residues
            or seq_len > max_seq_len
        )

        if flush:
            n_saved += _flush_batch(batch, model, tokenizer, device, embeddings_dir, stats)
            batch = []

    # Flush the remainder here rather than via a `seq_idx == n_total` condition
    # inside the loop. On a resumed run the final sequences are typically already
    # cached, and `continue` skips the flush check -- so an end-of-loop condition
    # silently dropped the whole trailing batch while still reporting success.
    if batch:
        n_saved += _flush_batch(batch, model, tokenizer, device, embeddings_dir, stats)
        batch = []

    if stats['bad_seqs']:
        print(f"WARNING: {stats['bad_seqs']} sequence(s) contained "
              f"{stats['bad_values']} non-finite value(s), zeroed on save. "
              'Run preflight.py before training, and regenerate in full precision '
              'if this was a --half run.')
    print(f'Done. {n_saved} embeddings saved to {embeddings_dir}')


def main():
    parser = argparse.ArgumentParser(
        description='Generate ProstT5 per-residue embeddings. '
                    'Output files use the same MD5-hash naming as the ESM-2 script. '
                    'Pass --embedding_dim 1024 to the training script when using these.'
    )
    parser.add_argument('fasta_file', type=pathlib.Path, help='Input FASTA file')
    parser.add_argument('output_dir', type=pathlib.Path, help='Directory to write .pt embedding files')
    parser.add_argument('--model', type=str, default='Rostlab/ProstT5',
                        help='HuggingFace repo name or path to a local ProstT5 checkpoint')
    parser.add_argument('--half', action='store_true', help='Use fp16 (saves ~2x memory, minor accuracy trade-off)')
    parser.add_argument('--max_seq_len', type=int, default=1000, help='Sequences longer than this are processed alone')
    parser.add_argument('--max_residues', type=int, default=4000, help='Max total residues per batch')
    parser.add_argument('--max_batch', type=int, default=100, help='Max sequences per batch')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    generate_prost5_embeddings(
        args.fasta_file,
        args.output_dir,
        model_name=args.model,
        half_precision=args.half,
        max_seq_len=args.max_seq_len,
        max_residues=args.max_residues,
        max_batch=args.max_batch,
    )


if __name__ == '__main__':
    main()
