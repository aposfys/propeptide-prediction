'''
Generate ESM3 embeddings using every input track that can be derived from
sequence + predicted structure, i.e. every track that does NOT leak the label.

make_embeddings.py passes `sequence_tokens=` alone, so all of ESM3's other
tracks are absent and the model runs sequence-only. The sequence-only ESM3
results therefore never tested whether multimodality helps -- the modalities
were never wired up, because the pipeline it replaced (ESM-2) has one input and
the dataset carries no 3D data.

Tracks used (all derived from the AlphaFold DB model, which is itself predicted
from sequence alone -- no annotation enters):

    sequence_tokens   always
    structure_coords  backbone coordinates, consumed by Geometric Attention
    structure_tokens  VQ-VAE tokens from atom37 coordinates
    sasa_tokens       ProteinChain.sasa()
    ss8_tokens        ProteinChain.dssp(), opt-in via --ss8 (needs mkdssp)

ESM3 has TWO independent structural pathways, and conditioning wants both
(Hayes et al. 2025, Science 387:850, supplementary A.1.5.1 / A.1.6):

    "Structure coordinates are parsed through the Geometric Attention and are
     not embedded."
    "Geometric Attention ... leverages fine-grained 3D information via
     conditioning on atomic coordinates of backbone atoms. Coordinates are only
     used as model inputs."
    "Structure Tokens ... enable faster learning due to rich local neighborhood
     semantics being compressed into tokens. Structure tokens are generally used
     as model outputs."

So tokens are the compressed/output representation and coordinates are the
fine-grained input-conditioning path. Passing tokens alone -- the obvious
reading of the SDK -- silently skips Geometric Attention entirely.

Tracks deliberately NOT used:

    function_tokens, residue_annotation_tokens
        The paper is explicit that these are annotation-derived: "Residue
        annotations: InterPro annotations are tokenized as a multi-hot feature
        vector (1478 dimensions) over possible InterPro labels." The labels here
        ARE UniProt PROPEP features, and InterPro annotates propeptide domains
        directly (e.g. "Peptidase inhibitor I9", the subtilisin propeptide) with
        their boundaries. Feeding these lets the model read the answer, and
        GraphPart does not protect against it -- it partitions by sequence
        identity, not by annotation source. It would also break the tool's
        purpose, which is predicting propeptides for unannotated sequences.

    per_res_plddt, average_plddt
        Available, and AFDB gives us real values, but the paper rules them out:
        "There are two additional tracks used during pretraining only: (h)
        per-residue confidence (pLDDT) and (i) averaged confidence (pLDDT). At
        inference time, these values are fixed, and these tracks are equivalent
        to adding a constant vector z_plddt." Feeding real pLDDT at inference
        adds no information and departs from the convention the model expects.
        --plddt exists to test that claim, and is off by default.

Other design decisions:

* Proteins without a usable structure are KEPT with the extra tracks masked.
  Dropping them would change the train/val/test partitions and break
  comparability with the sequence-only ESM3 and ESM-2 runs. labeled_sequences.csv
  and graphpart_assignments.csv are untouched -- GraphPart must NOT be re-run.
* Only exact sequence matches get a structure. A near match is a different
  isoform, and per-residue tracks aligned to the wrong residues are worse than
  none. This also handles AFDB's >2700aa fragmentation for free: the F1 fragment
  covers part of the sequence, fails the match, and is masked.
* Files are keyed by md5 of the sequence, matching make_embeddings.py and
  dataset.py. A fresh --out_dir is required: like the original, this skips
  hashes it already finds, so reusing a directory silently keeps old tensors.

GOTCHA worth knowing: ProteinChain.from_pdb() does not preserve the B-factor
column -- it returns a constant 1.0. AFDB stores per-residue pLDDT there, so
pLDDT is parsed from the PDB directly with biotite. Reading it off ProteinChain
would silently feed uniform confidence for every residue.
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


def _per_residue_plddt(pdb_path: str, n_res: int):
    '''Per-residue pLDDT from the PDB B-factor column, scaled to [0, 1].

    ProteinChain drops b_factor, so read the file directly.
    '''
    import numpy as np
    import biotite.structure.io.pdb as pdb
    try:
        arr = pdb.PDBFile.read(pdb_path).get_structure(model=1, extra_fields=['b_factor'])
        ca = arr[arr.atom_name == 'CA']
        if len(ca) != n_res:
            return None
        vals = np.asarray(ca.b_factor, dtype='float32')
        if vals.max() <= 1.01:
            return None          # constant/absent B-factors, not real pLDDT
        return torch.tensor(vals / 100.0, dtype=torch.float32)
    except Exception:
        return None


def _load_structure(pdb_path: str, sequence: str, want_ss8: bool):
    '''Return (coords, sasa, plddt, ss8) for `sequence`, or None if unusable.'''
    from esm.utils.structure.protein_chain import ProteinChain
    try:
        chain = ProteinChain.from_pdb(pdb_path)
    except Exception:
        return None
    if chain.sequence != sequence:
        return None

    coords = torch.tensor(chain.atom37_positions, dtype=torch.float32)

    try:
        sasa = [float(x) for x in chain.sasa()]
        if len(sasa) != len(sequence):
            sasa = None
    except Exception:
        sasa = None

    plddt = _per_residue_plddt(pdb_path, len(sequence))

    ss8 = None
    if want_ss8:
        try:
            ss8 = ''.join(chain.dssp())
            if len(ss8) != len(sequence):
                ss8 = None
        except Exception:
            ss8 = None

    return coords, sasa, plddt, ss8


def generate(data_file: str, structures_dir: str, out_dir: str, no_structure: bool,
             use_sasa: bool, use_plddt: bool, use_ss8: bool, limit: int) -> None:
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

    if use_ss8:
        import shutil
        if shutil.which('mkdssp') is None:
            print('WARNING: --ss8 requested but mkdssp is not on PATH. The ss8 '
                  'track will be masked for every protein. Install it with '
                  '`conda install -c conda-forge dssp`, or drop --ss8.')

    # Deduplicate by sequence hash, as make_embeddings.py does. Prefer an
    # accession that HAS a usable structure -- taking the first one blindly
    # would drop the structure track for any sequence whose first accession
    # happens to be the one AFDB is missing.
    best = {}
    for acc, seq in zip(df['protein_id'].astype(str), df['sequence'].astype(str)):
        h = hash_aa_string(seq)
        has_struct = manifest.get(acc, {}).get('status') == 'ok'
        if h not in best or (has_struct and not best[h][1]):
            best[h] = ((acc, seq, h), has_struct)
    records = [v[0] for v in best.values()]
    n_struct = sum(1 for v in best.values() if v[1])
    print(f'{len(records)} unique sequences to embed; {n_struct} have a usable '
          f'structure ({100*n_struct/max(1,len(records)):.1f}%)')

    if limit:
        records = records[:limit]
        print(f'--limit {limit}: smoke test, embedding {len(records)} sequences only')

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

            loaded = None
            if not no_structure and manifest.get(acc, {}).get('status') == 'ok':
                loaded = _load_structure(os.path.join(structures_dir, f'{acc}.pdb'),
                                         seq, use_ss8)
                if loaded is None:
                    stats['parse_failed'] += 1

            if loaded is None:
                protein = ESMProtein(sequence=seq)
                plddt = None
            else:
                coords, sasa, plddt, ss8 = loaded
                protein = ESMProtein(
                    sequence=seq,
                    coordinates=coords,
                    sasa=sasa if (use_sasa and sasa is not None) else None,
                    secondary_structure=ss8 if (use_ss8 and ss8 is not None) else None,
                )
                if not use_plddt:
                    plddt = None
                stats['with_structure'] += 1
                if use_sasa and sasa is not None:
                    stats['with_sasa'] += 1
                if use_ss8 and ss8 is not None:
                    stats['with_ss8'] += 1
                if plddt is not None:
                    stats['with_plddt'] += 1

            encoded = esm_model.encode(protein)
            seq_tokens = encoded.sequence.unsqueeze(0).to(device)
            kwargs = {'sequence_tokens': seq_tokens}

            # `coordinates` -> structure_coords feeds Geometric Attention, which
            # is a separate pathway from the embedded structure tokens. Passing
            # tokens alone skips it and loses the fine-grained 3D conditioning.
            for field, arg in (('coordinates', 'structure_coords'),
                               ('structure', 'structure_tokens'),
                               ('sasa', 'sasa_tokens'),
                               ('secondary_structure', 'ss8_tokens')):
                tok = getattr(encoded, field, None)
                if tok is not None:
                    kwargs[arg] = tok.unsqueeze(0).to(device)
                    stats[f'track_{arg}'] += 1

            if plddt is not None:
                # Match the token length exactly (BOS/EOS get 0).
                full = torch.zeros(seq_tokens.shape[-1], dtype=torch.float32)
                full[1:1 + plddt.shape[0]] = plddt
                kwargs['per_res_plddt'] = full.unsqueeze(0).to(device)

            if loaded is None:
                stats['sequence_only'] += 1

            out = esm_model(**kwargs)

            # Same LayerNorm fix as make_embeddings.py: ESMOutput.embeddings is
            # the raw pre-norm residual stream (~9800 per-token L2) because
            # TransformerStack.forward returns `self.norm(x), x, hiddens` and
            # ESM3.forward unpacks it as `x, embedding, _`. The model's own heads
            # consume the normalised tensor and LSTMCNN has no input
            # normalisation. The norm is per-token, so it commutes with the slice.
            emb = esm_model.transformer.norm(out.embeddings)[0, 1:-1].cpu()

            if emb.shape[0] != len(seq):
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
                   help='Directory written by fetch_afdb_structures.py.')
    p.add_argument('--out_dir', required=True,
                   help='Fresh output directory. Existing hashes are skipped, so '
                        'reusing a sequence-only directory writes nothing.')
    p.add_argument('--no_structure', action='store_true',
                   help='Ablation: run this exact code path with every extra '
                        'track masked. Isolates the track contribution from any '
                        'other difference between this script and the original.')
    p.add_argument('--no_sasa', action='store_true', help='Disable the SASA track.')
    p.add_argument('--plddt', action='store_true',
                   help='Feed real per-residue pLDDT. OFF by default: the ESM3 '
                        'paper states the pLDDT tracks are used during '
                        'pretraining only and are "equivalent to adding a '
                        'constant vector" at inference, so this adds no '
                        'information and departs from the expected convention. '
                        'Kept only so the claim can be tested.')
    p.add_argument('--ss8', action='store_true',
                   help='Enable the ss8 track. Requires mkdssp on PATH '
                        '(conda install -c conda-forge dssp); warns and masks if absent.')
    p.add_argument('--limit', type=int, default=0,
                   help='Embed only the first N sequences. Use this to smoke-test '
                        'the track plumbing before committing to the full set.')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    generate(args.data_file, args.structures_dir, args.out_dir, args.no_structure,
             not args.no_sasa, args.plddt, args.ss8, args.limit)


if __name__ == '__main__':
    main()
