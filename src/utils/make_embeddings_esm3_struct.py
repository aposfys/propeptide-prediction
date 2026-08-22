'''
Generate ESM3 embeddings with BOTH the sequence and structure tracks.

This is the multimodal counterpart to make_embeddings.py. That extractor passes
`sequence_tokens=` alone, so ESM3's other tracks are absent and the model runs
sequence-only -- meaning the sequence-only ESM3 results never tested whether
multimodality helps. This one adds structure tokens derived from AlphaFold DB
predictions (fetch them first with fetch_afdb_structures.py).

Design decisions worth knowing:

* Proteins with no usable structure are KEPT, with the structure track left
  masked. Dropping them would change the train/val/test partitions and make the
  result incomparable to the sequence-only and ESM-2 runs. The whole point is a
  matched comparison, so the protein set must stay identical.
* Only exact sequence matches get a structure. A near match is a different
  isoform, and per-residue structure tokens aligned to the wrong residues are
  worse than no structure.
* The FUNCTION track is deliberately NOT used. Function tokens derive from
  InterPro/UniProt annotations and the labels here ARE UniProt propeptide
  annotations -- that is label leakage. Structure tokens from coordinates are
  safe; annotation-derived tracks are not.
* Files are keyed by md5 of the sequence, matching make_embeddings.py and
  dataset.py. Writing to a fresh --out_dir is required: like the original, this
  skips hashes it already finds, so reusing a directory silently keeps the old
  sequence-only tensors.
'''
import argparse
import json
import os
from collections import Counter
from hashlib import md5

import pandas as pd
import torch
from tqdm.auto import tqdm


def hash_aa_string(string: str) -> str:
    return md5(string.encode()).digest().hex()


def _load_structure(pdb_path: str, sequence: str):
    '''Return atom37 coordinates for `sequence`, or None if unusable.'''
    from esm.utils.structure.protein_chain import ProteinChain
    try:
        chain = ProteinChain.from_pdb(pdb_path)
    except Exception:
        return None
    if chain.sequence != sequence:
        return None
    return torch.tensor(chain.atom37_positions, dtype=torch.float32)


def generate(data_file: str, structures_dir: str, out_dir: str,
             no_structure: bool = False) -> None:
    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESMProtein

    df = pd.read_csv(data_file)

    manifest = {}
    manifest_path = os.path.join(structures_dir, 'manifest.json')
    if os.path.isfile(manifest_path):
        manifest = json.load(open(manifest_path))
        usable = sum(1 for v in manifest.values() if v.get('status') == 'ok')
        print(f'manifest: {usable}/{len(manifest)} accessions have a usable structure')
    elif not no_structure:
        raise SystemExit(
            f'No manifest at {manifest_path}. Run fetch_afdb_structures.py first, '
            'or pass --no_structure to reproduce the sequence-only baseline.'
        )

    # Deduplicate by sequence hash, as make_embeddings.py does: several
    # accessions can share a sequence and the cache is keyed by hash. Prefer an
    # accession that HAS a usable structure -- picking the first one blindly
    # would silently drop the structure track for any sequence whose first
    # accession happens to be the one AFDB is missing.
    best, records = {}, []
    for acc, seq in zip(df['protein_id'].astype(str), df['sequence'].astype(str)):
        h = hash_aa_string(seq)
        has_struct = manifest.get(acc, {}).get('status') == 'ok'
        if h not in best or (has_struct and not best[h][1]):
            best[h] = ((acc, seq, h), has_struct)
    records = [v[0] for v in best.values()]
    n_struct = sum(1 for v in best.values() if v[1])
    print(f'{len(records)} unique sequences to embed; '
          f'{n_struct} have a usable structure ({100*n_struct/max(1,len(records)):.1f}%)')

    esm_model = ESM3.from_pretrained('esm3_sm_open_v1').eval()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    esm_model = esm_model.to(device)
    print(f'Embedding on {device}')

    stats = Counter()

    with torch.no_grad():
        for acc, seq, h in tqdm(records):
            out_path = os.path.join(out_dir, f'{h}.pt')
            if os.path.isfile(out_path):
                stats['cached'] += 1
                continue

            coords = None
            if not no_structure and manifest.get(acc, {}).get('status') == 'ok':
                coords = _load_structure(os.path.join(structures_dir, f'{acc}.pdb'), seq)
                if coords is None:
                    stats['parse_failed'] += 1

            protein = ESMProtein(sequence=seq) if coords is None else \
                ESMProtein(sequence=seq, coordinates=coords)
            encoded = esm_model.encode(protein)

            kwargs = {'sequence_tokens': encoded.sequence.unsqueeze(0).to(device)}
            structure_tokens = getattr(encoded, 'structure', None)
            if structure_tokens is not None:
                kwargs['structure_tokens'] = structure_tokens.unsqueeze(0).to(device)
                stats['with_structure'] += 1
            else:
                # No structure track: ESM3 masks it internally, which is exactly
                # what make_embeddings.py does for every protein.
                stats['sequence_only'] += 1

            out = esm_model(**kwargs)

            # Same LayerNorm fix as make_embeddings.py: ESMOutput.embeddings is
            # the raw pre-norm residual stream (~9800 per-token L2) because
            # TransformerStack.forward returns `self.norm(x), x, hiddens` and
            # ESM3.forward unpacks it as `x, embedding, _`. The model's own heads
            # consume the normalised tensor, and LSTMCNN has no input
            # normalisation. The norm is per-token, so it commutes with the
            # BOS/EOS slice.
            emb = esm_model.transformer.norm(out.embeddings)[0, 1:-1].cpu()

            if emb.shape[0] != len(seq):
                # A length mismatch means the tracks disagree about the residue
                # count, which would misalign every label downstream.
                raise RuntimeError(
                    f'{acc}: embedding length {emb.shape[0]} != sequence length '
                    f'{len(seq)}. Refusing to write a misaligned tensor.'
                )

            torch.save(emb, out_path)

    print('\n=== extraction summary ===')
    for k, v in stats.most_common():
        print(f'  {k:18} {v:6}')
    total = stats['with_structure'] + stats['sequence_only']
    if total:
        print(f'\n  {100*stats["with_structure"]/total:.1f}% of newly embedded '
              f'sequences carry a real structure track.')
    if stats['with_structure'] == 0 and not no_structure:
        print('\n  WARNING: no sequence got a structure track. This output is '
              'identical to the sequence-only baseline -- check the manifest.')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data_file', default='data/labeled_sequences.csv')
    p.add_argument('--structures_dir', required=True,
                   help='Directory written by fetch_afdb_structures.py (.pdb + manifest.json).')
    p.add_argument('--out_dir', required=True,
                   help='Fresh output directory. Existing hashes are skipped, so '
                        'reusing a sequence-only directory writes nothing.')
    p.add_argument('--no_structure', action='store_true',
                   help='Ablation: run this exact code path with the structure '
                        'track masked for every protein. Isolates the structure '
                        'contribution from any other difference in this script.')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    generate(args.data_file, args.structures_dir, args.out_dir, args.no_structure)


if __name__ == '__main__':
    main()
