'''
Repair ESM3 embeddings that were saved from `ESMOutput.embeddings`.

Background. ESM3's TransformerStack.forward returns `self.norm(x), x, hiddens`,
and ESM3.forward unpacks that as `x, embedding, _` before handing `embedding` to
OutputHeads as `ESMOutput.embeddings`. So `.embeddings` is the RAW pre-LayerNorm
residual stream, not the tensor the model's own heads consume.

fair-esm does the opposite for ESM-2: it applies `emb_layer_norm_after` and then
overwrites `representations[33]` with the normalised tensor. So an ESM-2 baseline
trains on LayerNorm-ed features while ESM3 does not, and a comparison between the
two measures normalisation rather than the embedder.

Measured per-token L2 norm on this dataset:
    ESM-2 L33                    10.12
    ESM3 after transformer.norm  11.62
    ESM3 .embeddings (raw)     9792.73     <- ~840x too large

Why this does not need an ESM3 re-run: the final norm is a per-token LayerNorm
over the feature dimension, so it commutes with the [1:-1] BOS/EOS slice that
make_embeddings.py already applied. Renormalising the saved tensors is therefore
exact with respect to what is on disk.

Note on precision: if the embeddings were written in bfloat16, this repair is
exact w.r.t. the stored values but still carries bf16's ~0.4% relative error. For
a thesis-facing artefact, re-extracting with the fixed make_embeddings.py is
cleaner.

Usage:
    python -m src.utils.renorm_esm3_embeddings \
        /data/apostolos/embeddings/esm3 \
        /data/apostolos/embeddings/esm3_normed
'''
import argparse
import os
import pathlib

import torch
from tqdm.auto import tqdm


def main():
    p = argparse.ArgumentParser()
    p.add_argument('src_dir', type=pathlib.Path,
                   help='directory of .pt files written from ESMOutput.embeddings')
    p.add_argument('dst_dir', type=pathlib.Path,
                   help='output directory (must differ from src_dir)')
    p.add_argument('--dtype', default='float32', choices=['float32', 'bfloat16'])
    args = p.parse_args()

    if args.src_dir.resolve() == args.dst_dir.resolve():
        raise SystemExit('src_dir and dst_dir must differ — this is not an in-place fix.')

    os.makedirs(args.dst_dir, exist_ok=True)

    # Only the final LayerNorm is needed. Load on CPU and drop the rest of the model.
    from esm.models.esm3 import ESM3
    model = ESM3.from_pretrained('esm3_sm_open_v1').eval()
    norm = model.transformer.norm.to(torch.float32)
    del model

    files = sorted(args.src_dir.glob('*.pt'))
    if not files:
        raise SystemExit(f'no .pt files in {args.src_dir}')
    print(f'{len(files)} embeddings to renormalise')

    out_dtype = getattr(torch, args.dtype)
    checked = False
    with torch.no_grad():
        for f in tqdm(files):
            x = torch.load(f, map_location='cpu').to(torch.float32)

            if not checked:
                # Pre-norm ESM3 features have per-token norms in the thousands.
                # If these are already normalised (~10), refuse rather than
                # double-normalise, which would silently corrupt them.
                n = x.norm(dim=-1).mean().item()
                print(f'  first file: mean per-token |x| = {n:.1f}')
                if n < 500:
                    raise SystemExit(
                        'Refusing to run: these do not look like pre-norm embeddings '
                        f'(mean |x| = {n:.1f}, expected >5000). Double-normalising '
                        'would corrupt them.'
                    )
                checked = True

            torch.save(norm(x).to(out_dtype), args.dst_dir / f.name)

    print(f'Done -> {args.dst_dir}')
    print('Point --embeddings_dir at the new directory. Run preflight.py first; '
          'its scale check should now report ~0.3 x sqrt(dim).')


if __name__ == '__main__':
    main()
