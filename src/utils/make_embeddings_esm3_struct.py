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


def _parse_layers(args) -> list:
    """Resolve --layers / --layer into a list of 1-indexed blocks, or [] for the default.

    --layer is kept so the commands already in RESULTS.md keep working; --layers
    supersedes it when both are given.
    """
    if getattr(args, 'layers', ''):
        return [int(x) for x in str(args.layers).replace(' ', '').split(',') if x]
    return [args.layer] if getattr(args, 'layer', 0) else []


def generate(data_file: str, structures_dir: str, out_dir: str, no_structure: bool,
             use_sasa: bool, use_plddt: bool, use_ss8: bool, limit: int,
             max_struct_len: int = 0, gpu_max_len: int = 0, layers: list | None = None) -> None:
    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESMProtein

    # Record what produced this directory. Which tracks were fed is not
    # recoverable from the .pt files, and a run's provenance must not depend on
    # remembering the command line -- the same lesson as config.json:label_type.
    json.dump(
        {'no_structure': no_structure, 'use_sasa': use_sasa, 'use_plddt': use_plddt,
         'use_ss8': use_ss8, 'max_struct_len': max_struct_len,
         'gpu_max_len': gpu_max_len, 'layers': layers,
         'structures_dir': structures_dir, 'data_file': data_file},
        open(os.path.join(out_dir, 'extraction_config.json'), 'w'), indent=2,
    )

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

    # Force float32. Two reasons, both load-bearing:
    #
    # 1. COMPARABILITY. On CUDA the SDK loads ESM3 in bfloat16; on CPU it loads
    #    float32. The sequence-only baseline (embeddings/esm3_normed) was
    #    extracted on a CPU-only node and is therefore float32. Leaving the GPU
    #    run in bf16 would change the numerical precision at the same time as
    #    adding the structure track, so any difference in the result could not be
    #    attributed to structure.
    #
    # 2. IT CRASHES. With structure conditioning the encoder reaches
    #    esm3.py:116, plddt_projection(rbf_16_fn(average_plddt)), where the SDK's
    #    default average_plddt is float32 while the projection weights are
    #    bf16 -> "mat1 and mat2 must have the same dtype, but got Float and
    #    BFloat16". Casting the whole model sidesteps it rather than patching one
    #    tensor and waiting for the next mismatch.
    #
    # Cost is roughly double the activation memory, which matters because
    # Geometric Attention is O(L^2) -- pair this with --max_struct_len.
    if device.type == 'cuda':
        esm_model = esm_model.to(torch.float32)
    print(f'Embedding on {device}, dtype {next(esm_model.parameters()).dtype}')

    # Intermediate-layer extraction.
    #
    # WHY: upstream TUNED ESM-2's layer -- they swept it and chose 33 of 33
    # (btad616 Fig. S9; their CSV has L32 at 0.494-0.542 vs L33 at 0.503-0.564).
    # ESM3 has only ever been read at layer 48 of 48, chosen by analogy rather
    # than measured. That is the one respect in which ESM-2 received treatment
    # ESM3 did not, so it is the only lever that can narrow the gap rather than
    # lift both models.
    #
    # It is also plausible on mechanism: ESM3's final layers specialise toward
    # its generative objective -- the authors' own explanation for weaker
    # representations -- so mid-network features may suit a token-level task
    # better.
    #
    # HOW: ESM3.forward calls `x, embedding, _ = self.transformer(...)`, throwing
    # the per-block hidden states away. A forward hook on the transformer catches
    # the third return value without duplicating any of forward()'s token
    # defaulting, so every track behaves exactly as it does normally.
    #
    # hiddens[i] is the output of block i+1, so layer L is hiddens[L-1] and
    # hiddens[-1] is the pre-norm final. --layer 48 therefore reproduces the
    # default path exactly, which is a free correctness check.
    #
    # The same transformer.norm is applied whichever layer is taken. It is not
    # trained for intermediate layers, but it is a per-token LayerNorm, so it
    # standardises scale and keeps every layer's output in the band preflight
    # expects (~0.3 x sqrt(dim)). Without it the pre-norm residual stream is the
    # ~840x off-scale tensor that caused the original bug.
    captured = {}

    def _grab_hiddens(_module, _inputs, output):
        captured['hiddens'] = output[2]

    hook = None
    if layers:
        n_blocks = len(esm_model.transformer.blocks)
        for l in layers:
            if not 1 <= l <= n_blocks:
                raise SystemExit(f'--layers values must be in 1..{n_blocks}, got {l}')
        hook = esm_model.transformer.register_forward_hook(_grab_hiddens)
        # Concatenating several layers is standard practice for protein LMs: the
        # blocks are not redundant, and a readout head can weight them itself.
        # Layers 44 and 48 score equivalently here (val 0.7064 / 0.7045) but are
        # not identical, so they may carry complementary signal.
        if len(layers) == 1:
            l = layers[0]
            print(f'Extracting layer {l} of {n_blocks} '
                  f'({"final -- identical to the default path" if l == n_blocks else "intermediate"})')
        else:
            print(f'Concatenating layers {layers} of {n_blocks} '
                  f'-> {1536 * len(layers)}-dim (pass --embedding_dim {1536 * len(layers)} to run.py)')

    stats = Counter()

    # Geometric Attention lives in the FIRST transformer block and runs
    # unconditionally -- the ESM3 supplementary notes that "partially or fully
    # masked coordinates can be input", so the L x L x heads x 3 pairwise tensor
    # is allocated whether or not coordinates are passed. At 256 v-heads that is
    # 37.6 GiB for a ~3600-residue protein, which OOMs a 31 GiB card regardless
    # of --max_struct_len. The sequence-only extraction only ever survived it
    # because the CPU node had 251 GB of system RAM.
    #
    # So the long sequences are deferred to a second pass with the model moved
    # to CPU: slow, but only a handful of proteins, and CPU gives the same
    # float32 result so the output stays homogeneous.
    if gpu_max_len > 0 and device.type == 'cuda':
        fast = [r for r in records if len(r[1]) <= gpu_max_len]
        slow = [r for r in records if len(r[1]) > gpu_max_len]
    else:
        fast, slow = records, []
    if slow:
        print(f'{len(slow)} sequences longer than {gpu_max_len} residues will be '
              f'embedded on CPU in a second pass.')

    def _embed_all(record_list, dev):
        for acc, seq, h in tqdm(record_list):
            out_path = os.path.join(out_dir, f'{h}.pt')
            if os.path.isfile(out_path):
                stats['cached'] += 1
                continue

            loaded = None
            # Geometric Attention is O(L^2) in memory: a 3971-residue protein
            # asks for a single 37.6 GiB allocation and OOMs a 31 GiB card.
            # Above the cap the structural tracks are dropped and the protein
            # falls back to sequence-only, exactly like the ~1.3% with no AFDB
            # model. 1024 matches ESM3's own prompt-following evaluation, which
            # "removes proteins with length greater than 1024".
            too_long = max_struct_len > 0 and len(seq) > max_struct_len
            if too_long:
                stats['skipped_too_long'] += 1
            elif not no_structure and manifest.get(acc, {}).get('status') == 'ok':
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
            seq_tokens = encoded.sequence.unsqueeze(0).to(dev)
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
                    kwargs[arg] = tok.unsqueeze(0).to(dev)
                    stats[f'track_{arg}'] += 1

            if plddt is not None:
                # Match the token length exactly (BOS/EOS get 0).
                full = torch.zeros(seq_tokens.shape[-1], dtype=torch.float32)
                full[1:1 + plddt.shape[0]] = plddt
                kwargs['per_res_plddt'] = full.unsqueeze(0).to(dev)

            if loaded is None:
                stats['sequence_only'] += 1

            out = esm_model(**kwargs)

            # Same LayerNorm fix as make_embeddings.py: ESMOutput.embeddings is
            # the raw pre-norm residual stream (~9800 per-token L2) because
            # TransformerStack.forward returns `self.norm(x), x, hiddens` and
            # ESM3.forward unpacks it as `x, embedding, _`. The model's own heads
            # consume the normalised tensor and LSTMCNN has no input
            # normalisation. The norm is per-token, so it commutes with the slice.
            # .float() is a belt-and-braces guard: dataset.py and preflight.py
            # both expect float32, and a half-precision tensor here would only
            # surface much later as a dtype error during training.
            if layers:
                # Norm each layer separately before concatenating, so every block
                # lands in the same scale band. Sharing one norm across the
                # concatenation would let a single layer dominate.
                parts = [esm_model.transformer.norm(captured['hiddens'][l - 1])[0, 1:-1]
                         for l in layers]
                emb = torch.cat(parts, dim=-1).float().cpu()
            else:
                emb = esm_model.transformer.norm(out.embeddings)[0, 1:-1].float().cpu()

            if emb.shape[0] != len(seq):
                raise RuntimeError(
                    f'{acc}: embedding length {emb.shape[0]} != sequence length '
                    f'{len(seq)}. Refusing to write a misaligned tensor.'
                )

            torch.save(emb, out_path)

    with torch.no_grad():
        _embed_all(fast, device)
        if slow:
            # One transfer, not one per sequence: move the model to CPU once and
            # run the whole long-sequence tail there.
            print(f'\nMoving model to CPU for {len(slow)} long sequences '
                  f'(expect minutes each, not seconds).')
            esm_model = esm_model.to('cpu')
            torch.cuda.empty_cache()
            _embed_all(slow, torch.device('cpu'))

    if hook is not None:
        hook.remove()

    print('\n=== extraction summary ===')
    for k, v in stats.most_common():
        print(f'  {k:18} {v:6}')
    total = stats['with_structure'] + stats['sequence_only']
    if total:
        print(f'\n  {100*stats["with_structure"]/total:.1f}% of newly embedded '
              f'sequences carry a real structure track.')
    elif stats['cached']:
        # Nothing was embedded this run because every hash was already on disk.
        # That is the normal resume path, not a failure -- do not warn.
        print(f'\n  Nothing to do: all {stats["cached"]} sequences were already '
              f'in --out_dir. Use a fresh directory to re-extract.')
    if total and stats['with_structure'] == 0 and not no_structure:
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
    p.add_argument('--max_struct_len', type=int, default=0,
                   help='Drop the structural tracks for sequences longer than N '
                        'residues; they fall back to sequence-only. Geometric '
                        'Attention is O(L^2): a 3971-residue protein requests a '
                        'single 37.6 GiB allocation. 1024 is the recommended '
                        'value and matches ESM3\'s own prompt-following '
                        'evaluation. 0 = no cap.')
    p.add_argument('--gpu_max_len', type=int, default=0,
                   help='Sequences longer than N residues are embedded on CPU in '
                        'a second pass instead of the GPU. Geometric Attention '
                        'runs unconditionally in block 1 and allocates an '
                        'L x L x heads x 3 tensor -- 37.6 GiB at L~3600 -- so '
                        'long proteins OOM a 31 GiB card whether or not '
                        'structure is passed. CPU gives the same float32 result. '
                        '2000 defers 15 of 8449 sequences (0.18%%). 0 = off.')
    p.add_argument('--layers', type=str, default='',
                   help='Comma-separated transformer blocks to extract, 1-indexed, '
                        'e.g. "48" or "44,48". Empty = the default final output. '
                        'Several layers are normed individually and concatenated, '
                        'giving 1536 x N dims -- pass the matching --embedding_dim '
                        'to run.py. ESM3-small has 48 blocks; "48" reproduces the '
                        'default path exactly.')
    p.add_argument('--layer', type=int, default=0,
                   help='Extract from transformer block N (1-indexed) instead of '
                        'the final output. 0 = default/final, identical to the '
                        'previous behaviour. ESM3-small has 48 blocks, so '
                        '--layer 48 reproduces the default exactly. Upstream '
                        'tuned ESM-2\'s layer (Fig. S9, 33 of 33); ESM3\'s was '
                        'never measured, which is the one fairness gap left in '
                        'the comparison. transformer.norm is applied whichever '
                        'layer is taken, to keep the scale in the band preflight '
                        'expects.')
    p.add_argument('--limit', type=int, default=0,
                   help='Embed only the first N sequences. Use this to smoke-test '
                        'the track plumbing before committing to the full set.')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    generate(args.data_file, args.structures_dir, args.out_dir, args.no_structure,
             not args.no_sasa, args.plddt, args.ss8, args.limit, args.max_struct_len,
             args.gpu_max_len, _parse_layers(args))


if __name__ == '__main__':
    main()
