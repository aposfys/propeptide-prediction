'''
LoRA fine-tuning entry point for the propeptide CRF.

Deliberately SEPARATE from run.py / train_loop_crf.py. That pipeline produced
every number in RESULTS.md, and this project has already been bitten twice by
changes that altered a shared path silently -- `--patience` accepted and
ignored on two branches, and the recipe drift between branches. An
experimental path gets its own entry point rather than a branch inside the one
that works.

    python finetune.py --lora_blocks 12 --lora_lr 1e-4 --out_dir results/ft_r8_b12

Run the frozen control FIRST:

    python finetune.py --lora_blocks 0 --out_dir results/ft_frozen_control

That is the same crop, the same tokenisation and the same code path with no
adapters, so it isolates the rewrite from the adaptation. It should land near
the replicate mean of 0.5203 +/- 0.0185 (n=15), NOT the 0.5547 headline -- that
figure sits above the maximum of fifteen replicates of its own configuration
and is a fortunate draw. A control at ~0.52 is a pass; a control at 0.5547
would be suspicious.
'''
import argparse
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.models import LSTMCNNCRF
from src.models.plm_backbone import PLMBackbone, param_groups, summarise
from src.utils.finetune_glue import (SequenceCRFDataset, LengthBucketSampler,
                                     FineTunedCRF)
from src.utils.manuscript_metrics import compute_all_metrics


def parse_arguments(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # data
    p.add_argument('--data_file', default='data/labeled_sequences.csv')
    p.add_argument('--partitioning_file', default='data/graphpart_assignments.csv')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--max_len', type=int, default=2048,
                   help='Crop length. 2048 leaves ZERO unreachable propeptides on '
                        'valid and test; 1024 costs -0.0027 F1 and preferentially '
                        'deletes C-terminal propeptides of long precursors.')
    # PEFT
    p.add_argument('--model_name', default='esm3_sm_open_v1')
    p.add_argument('--lora_blocks', type=int, default=12,
                   help='Adapt the last N transformer blocks. 0 = frozen control.')
    p.add_argument('--lora_r', type=int, default=8)
    p.add_argument('--lora_alpha', type=int, default=16)
    p.add_argument('--lora_dropout', type=float, default=0.05)
    p.add_argument('--lora_lr', type=float, default=1e-4,
                   help='Two orders below the head LR. Running adapters at head LR '
                        'destroys pretrained features within a few hundred steps.')
    p.add_argument('--head_only_epochs', type=int, default=3,
                   help='Train the head alone first. Backpropagating a randomly '
                        'initialised head into a pretrained encoder from step 0 is '
                        'the classic way to wreck it.')
    p.add_argument('--no_grad_checkpoint', action='store_true')
    p.add_argument('--no_input_norm', action='store_true',
                   help='Disable the LayerNorm at the head input. Only for ablating '
                        'it -- without it the head is scale-sensitive, which is what '
                        'made the pre-norm embedding bug catastrophic.')
    # head (defaults are the tuned T4 configuration)
    p.add_argument('--lr', type=float, default=5.5e-3, help='Head learning rate.')
    p.add_argument('--dropout', type=float, default=0.6902)
    p.add_argument('--conv_dropout', type=float, default=0.5492)
    p.add_argument('--kernel_size', type=int, default=5)
    p.add_argument('--num_filters', type=int, default=48)
    p.add_argument('--hidden_size', type=int, default=48)
    p.add_argument('--use_focal', action='store_true')
    # optimisation
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--patience', type=int, default=15, help='0 = disabled.')
    p.add_argument('--batch_size', type=int, default=8, help='Real batch size.')
    p.add_argument('--accum_steps', type=int, default=9,
                   help='Gradient accumulation. 8 x 9 = 72, close to the batch 70 '
                        'that stabilised ESM3 (divergence at epoch 22 vs 6).')
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--warmup_frac', type=float, default=0.05,
                   help='Linear warmup as a fraction of total optimiser steps. A '
                        'deviation from upstream, applying to fine-tuned runs only.')
    p.add_argument('--clip', type=float, default=1.0)
    p.add_argument('--no_autocast', action='store_true')
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--num_cpu_threads', type=int, default=0)
    p.add_argument('--seed', type=int, default=42,
                   help='Fine-tuning adds encoder initialisation noise on top of '
                        'the head\'s, so unlike the frozen runs this path seeds and '
                        'records the seed.')
    return p.parse_args(argv)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(args):
    '''Train uses length bucketing; valid and test must NOT.

    src/train_loop_crf.py states the invariant: compute_all_metrics receives
    predictions in LOADER order and names/data in DATASET order, and pairs them
    positionally. A bucketing sampler permutes the loader order, so using it on
    valid or test silently mismatches every prediction with the wrong protein --
    no exception, just a wrong number.
    '''
    sets = {
        'train': SequenceCRFDataset(args.data_file, args.partitioning_file, (0, 1, 2), args.max_len),
        'valid': SequenceCRFDataset(args.data_file, args.partitioning_file, (3,), args.max_len),
        'test': SequenceCRFDataset(args.data_file, args.partitioning_file, (4,), args.max_len),
    }
    collate = SequenceCRFDataset.collate_fn

    train_loader = DataLoader(
        sets['train'],
        batch_sampler=LengthBucketSampler(sets['train'].lengths, args.batch_size, seed=args.seed),
        collate_fn=collate, num_workers=args.num_workers,
    )
    eval_loaders = {
        k: DataLoader(sets[k], batch_size=args.batch_size, shuffle=False,
                      collate_fn=collate, num_workers=max(0, args.num_workers - 1))
        for k in ('valid', 'test')
    }
    return sets, train_loader, eval_loaders


def build_model(args, device) -> FineTunedCRF:
    backbone = PLMBackbone(
        model_name=args.model_name,
        n_lora_blocks=args.lora_blocks,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        grad_checkpoint=not args.no_grad_checkpoint,
        max_len=args.max_len,
        device=device,
    )
    head = LSTMCNNCRF(
        input_size=backbone.d_model,
        num_labels=2,
        num_states=51,
        dropout_input=args.dropout,
        n_filters=args.num_filters,
        hidden_size=args.hidden_size,
        filter_size=args.kernel_size,
        dropout_conv1=args.conv_dropout,
    )
    return FineTunedCRF(backbone, head,
                        normalize_input=not args.no_input_norm,
                        autocast_backbone=not args.no_autocast).to(device)


def set_adapters_trainable(model: FineTunedCRF, flag: bool) -> int:
    n = 0
    for name, prm in model.named_parameters():
        if 'lora_' in name:
            prm.requires_grad_(flag)
            n += 1
    return n


def lr_lambda_factory(total_steps: int, warmup_frac: float):
    warmup = max(1, int(total_steps * warmup_frac))

    def fn(step: int) -> float:
        return min(1.0, (step + 1) / warmup)
    return fn


@torch.no_grad()
def evaluate(model: FineTunedCRF, loader: DataLoader, dataset: SequenceCRFDataset):
    model.eval()
    preds = []
    for sequences, _labels, _prop in loader:
        _probs, paths, _ = model(sequences, None, skip_marginals=True)
        preds.extend(paths)
    assert len(preds) == len(dataset.names), (
        f'{len(preds)} predictions for {len(dataset.names)} proteins -- loader '
        'order and dataset order have diverged.'
    )
    return compute_all_metrics(None, preds, None, dataset.names, dataset.data, windows=[3])[0]


def main() -> None:
    args = parse_arguments()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.num_cpu_threads:
        torch.set_num_threads(args.num_cpu_threads)
    set_seed(args.seed)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if device.type != 'cuda':
        raise SystemExit('Fine-tuning ESM3 on CPU is not viable. Run this on a GPU node.')

    sets, train_loader, eval_loaders = build_loaders(args)
    model = build_model(args, device)
    print(summarise(model))

    optimizer = torch.optim.AdamW(
        param_groups(model, head_lr=args.lr, lora_lr=args.lora_lr,
                     weight_decay=args.weight_decay)
    )
    steps_per_epoch = max(1, len(train_loader) // args.accum_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda_factory(steps_per_epoch * args.epochs, args.warmup_frac)
    )

    # Everything needed to reproduce or classify this run. RESULTS.md's "how runs
    # are classified" section exists because a config that omits a setting is a
    # config that lets it be silently ignored -- as --patience was.
    cfg = vars(args) | {
        'effective_batch': args.batch_size * args.accum_steps,
        'n_lora_params': model.backbone.n_lora_params,
        'n_geom_attn_disabled': model.backbone.n_geom_disabled,
        'd_model': model.backbone.d_model,
        'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'unreachable_segments': {k: v.n_unreachable for k, v in sets.items()},
    }
    json.dump(cfg, open(os.path.join(args.out_dir, 'config.json'), 'w'), indent=2)

    writer = SummaryWriter(args.out_dir)
    ckpt = os.path.join(args.out_dir, 'model.pt')
    best, patience_counter, step = -1.0, 0, 0

    for epoch in range(args.epochs):
        head_only = epoch < args.head_only_epochs and args.lora_blocks > 0
        if args.lora_blocks > 0:
            set_adapters_trainable(model, not head_only)

        model.train()
        losses = []
        optimizer.zero_grad(set_to_none=True)
        for i, (sequences, labels, _prop) in enumerate(train_loader):
            labels = labels.long().to(device)
            # skip_decode: training throws the Viterbi paths away, and the
            # backtrace is a per-timestep Python loop of .item() calls.
            _probs, _paths, loss = model(sequences, labels, skip_marginals=True,
                                         use_focal=args.use_focal, skip_decode=True)
            (loss / args.accum_steps).backward()
            losses.append(loss.item())
            if (i + 1) % args.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], args.clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

        vm = evaluate(model, eval_loaders['valid'], sets['valid'])
        score = vm['f1 propeptides']
        writer.add_scalar('Valid/f1_propeptides', score, epoch)
        writer.add_scalar('Train/loss', float(np.mean(losses)), epoch)

        improved = score > best
        if improved:
            best = score
            torch.save(model.state_dict(), ckpt)
            json.dump({**vm, 'epoch': epoch},
                      open(os.path.join(args.out_dir, 'valid_metrics.json'), 'w'), indent=2)
            patience_counter = 0
        else:
            patience_counter += 1

        tag = 'head-only' if head_only else 'adapters'
        pat = f'{patience_counter}/{args.patience}' if args.patience > 0 else 'off'
        print(f"  {'*' if improved else ' '} epoch {epoch+1:3d} [{tag}] "
              f"loss={np.mean(losses):.4f} val_f1={score:.4f} best={best:.4f} "
              f"lr={scheduler.get_last_lr()[0]:.2e} patience={pat}", flush=True)

        if args.patience > 0 and patience_counter >= args.patience:
            print(f'  Early stopping at epoch {epoch+1} (patience={args.patience}).')
            break

    model.load_state_dict(torch.load(ckpt, map_location=device))
    tm = evaluate(model, eval_loaders['test'], sets['test'])
    json.dump(tm, open(os.path.join(args.out_dir, 'test_metrics.json'), 'w'), indent=2)
    print('\n=== test ===')
    for k, v in tm.items():
        print(f'  {k:26} {v:.4f}')


if __name__ == '__main__':
    main()
