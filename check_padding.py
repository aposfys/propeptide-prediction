'''Two-minute padding sanity check -- no training, no adapters.

Run this FIRST if the frozen control's test F1 misses its band. A failure here
invalidates the --no_autocast and --no_input_norm ablations before you spend a
run on either, because both arms would be equally contaminated.

    python check_padding.py

fp32 by default: a failure then means a masking bug rather than bf16 reduction
noise. If it PASSES in fp32, that also bounds what --no_autocast can be worth --
the encoder's bf16 exposure is reduction-order noise, not a structural change.
'''
import argparse
import torch

from src.models.plm_backbone import PLMBackbone, check_batch_invariance
from src.utils.finetune_glue import SequenceCRFDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_file', default='data/labeled_sequences.csv')
    p.add_argument('--partitioning_file', default='data/graphpart_assignments.csv')
    p.add_argument('--max_len', type=int, default=2048)
    p.add_argument('--n', type=int, default=8, help='Sequences to sample.')
    p.add_argument('--atol', type=float, default=1e-4)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit('needs a GPU')

    ds = SequenceCRFDataset(args.data_file, args.partitioning_file, (4,), args.max_len)
    # Length-varied on purpose: uniform lengths would pad to nothing and the
    # check would pass trivially.
    order = sorted(range(len(ds)), key=lambda i: ds.lengths[i])
    picks = [order[int(k * (len(order) - 1) / (args.n - 1))] for k in range(args.n)]
    seqs = [ds.sequences[i][: args.max_len] for i in picks]
    print(f'  sampled lengths: {[len(s) for s in seqs]}')

    backbone = PLMBackbone(n_lora_blocks=0, grad_checkpoint=False,
                           max_len=args.max_len, device='cuda')
    r = check_batch_invariance(backbone, seqs, atol=args.atol)
    raise SystemExit(0 if r['ok'] else 1)


if __name__ == '__main__':
    main()
