'''
CRF train loop — propeptide cleavage site prediction via ESM3 + LSTM-CNN-CRF.

Training protocol (DeepPeptide, Bioinformatics 2023):
  - 50 epochs with patience-based early stopping (metric: propeptide F1).
  - 5-fold nested cross-validation: Optuna inner loop (4-fold) finds best
    hyperparameters; the 4 inner-fold models per outer fold are saved and
    used as a 5×4=20-model ensemble at inference.
  - ESM3 weights are frozen; only the prediction head is trained.
  - Loss: negative log-likelihood of the CRF (Viterbi / forward-backward).
'''
import json
import math
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import optuna
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .models import LSTMCNNCRF, SimpleLSTMCNNCRF, SelfAttentionCRF
from .utils.dataset import PrecomputedCSVForOverlapCRFDataset
from .utils.manuscript_metrics import compute_all_metrics

import argparse

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
global_step = 0


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get_dataloaders(
    args: argparse.Namespace,
    train_partitions: List[int],
    valid_partitions: List[int],
    test_partitions: List[int],
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    if args.embedding == 'precomputed':
        train_set = PrecomputedCSVForOverlapCRFDataset(
            args.embeddings_dir, args.data_file, args.partitioning_file,
            partitions=train_partitions,
        )
        valid_set = PrecomputedCSVForOverlapCRFDataset(
            args.embeddings_dir, args.data_file, args.partitioning_file,
            partitions=valid_partitions,
        )
        test_set = PrecomputedCSVForOverlapCRFDataset(
            args.embeddings_dir, args.data_file, args.partitioning_file,
            partitions=test_partitions,
        )
    else:
        raise NotImplementedError(args.embedding)

    print(
        f'Loaded data. {len(train_set)} train (p.{train_partitions}), '
        f'{len(valid_set)} valid (p.{valid_partitions}), '
        f'{len(test_set)} test (p.{test_partitions}).'
    )

    nw = getattr(args, 'num_workers', 2)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        collate_fn=train_set.collate_fn, num_workers=nw,
    )
    valid_loader = DataLoader(
        valid_set, batch_size=args.batch_size, shuffle=False,
        collate_fn=valid_set.collate_fn, num_workers=max(0, nw - 1),
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        collate_fn=test_set.collate_fn, num_workers=max(0, nw - 1),
    )
    return train_loader, valid_loader, test_loader


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def get_model(args: argparse.Namespace) -> torch.nn.Module:
    if args.model == 'lstmcnncrf':
        model = LSTMCNNCRF(
            input_size=args.embedding_dim,
            num_labels=2,
            dropout_input=args.dropout,
            num_states=51,
            n_filters=args.num_filters,
            hidden_size=args.hidden_size,
            filter_size=args.kernel_size,
            dropout_conv1=args.conv_dropout,
        )
    elif args.model == 'lstmcnncrf_simple':
        model = SimpleLSTMCNNCRF(
            input_size=args.embedding_dim,
            num_labels=2,
            dropout_input=args.dropout,
            num_states=51,
            n_filters=args.num_filters,
            hidden_size=args.hidden_size,
            filter_size=args.kernel_size,
            dropout_conv1=args.conv_dropout,
        )
    elif args.model == 'selfattentioncrf':
        model = SelfAttentionCRF(
            input_size=args.embedding_dim,
            hidden_size=args.hidden_size,
            num_labels=2,
            dropout_input=args.dropout,
            num_states=51,
            n_heads=args.num_filters,
            attn_dropout=args.conv_dropout,
        )
    else:
        raise NotImplementedError(args.model)

    print('Trainable params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model


def _flatten_lstm_if_present(model: torch.nn.Module) -> None:
    '''Call flatten_parameters only when the feature extractor has a biLSTM.'''
    fe = getattr(model, 'feature_extractor', None)
    lstm = getattr(fe, 'biLSTM', None)
    if lstm is not None:
        lstm.flatten_parameters()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_dataloader(
    loader: DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    writer: SummaryWriter,
    do_train: bool = True,
    use_focal: bool = False,
) -> Tuple[float, List[np.ndarray], List[List[int]], List[np.ndarray], List[np.ndarray]]:
    '''Run one epoch; collect per-sequence Viterbi paths, marginals, and labels.'''
    global global_step

    true_peptides = []
    labels = []
    probs = []
    preds = []
    epoch_loss = []

    model.train() if do_train else model.eval()

    for batch in loader:
        embeddings, mask, label, propeptides = batch
        embeddings = embeddings.to(device)
        mask = mask.to(device)
        label = label.to(device)

        if do_train:
            model.zero_grad()
            pos_probs, pos_preds, loss = model(
                embeddings, mask, label, skip_marginals=True, use_focal=use_focal,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            writer.add_scalar('Train/loss', loss.item(), global_step=global_step)
            global_step += 1
        else:
            with torch.no_grad():
                pos_probs, pos_preds, loss = model(
                    embeddings, mask, label, skip_marginals=True, use_focal=False,
                )

        true_peptides.extend(propeptides)
        probs.extend([pos_probs[i].detach().cpu().numpy() for i in range(pos_probs.shape[0])])
        labels.extend([label[i].detach().cpu().numpy() for i in range(label.shape[0])])
        preds.extend(pos_preds)
        epoch_loss.append(loss.item())

    return sum(epoch_loss) / len(epoch_loss), probs, preds, true_peptides, labels


def run_training_for_params(
    args: argparse.Namespace,
    model: torch.nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    writer: SummaryWriter,
    checkpoint_path: str,
) -> Tuple[Dict, float]:
    '''Train up to args.epochs with patience-based early stopping on propeptide F1.'''
    global global_step
    global_step = 0

    best_score = -1.0
    best_val_metrics = None
    patience_counter = 0
    use_focal = getattr(args, 'use_focal', True)

    # Linear warmup for the first warmup_epochs, then cosine decay to 1e-6.
    # Each inner fold gets a fresh scheduler because run_training_for_params is
    # called once per fold — no manual reset needed.
    base_lr = args.lr
    warmup_epochs = max(3, int(0.05 * args.epochs))

    def _lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = float(epoch - warmup_epochs) / float(max(1, args.epochs - warmup_epochs - 1))
        cosine_val = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return max(1e-6 / base_lr, cosine_val)

    scheduler = LambdaLR(optimizer, lr_lambda=_lr_lambda)

    for epoch in range(args.epochs):
        train_loss, _, _, _, _ = run_dataloader(
            train_loader, model, optimizer, writer, do_train=True, use_focal=use_focal,
        )

        _, valid_probs, valid_preds, _, valid_labels = run_dataloader(
            valid_loader, model, optimizer, writer, do_train=False,
        )
        valid_metrics = compute_all_metrics(
            valid_probs, valid_preds, valid_labels,
            valid_loader.dataset.names, valid_loader.dataset.data,
            windows=[3],
        )[0]

        score = valid_metrics['f1 propeptides']
        scheduler.step()
        writer.add_scalar('Valid/f1_propeptides', score, global_step=epoch)
        current_lr = optimizer.param_groups[0]['lr']

        improved = score > best_score
        if improved:
            best_score = score
            best_val_metrics = {**valid_metrics, 'epoch': epoch}
            torch.save(model.state_dict(), checkpoint_path)
            patience_counter = 0
        else:
            patience_counter += 1

        marker = '*' if improved else ' '
        print(
            f'  {marker} epoch {epoch+1:3d}  loss={train_loss:.4f}  '
            f'val_f1={score:.4f}  best={best_score:.4f}  '
            f'patience={patience_counter}/{args.patience}  lr={current_lr:.2e}',
            flush=True,
        )

        if patience_counter >= args.patience:
            print(f'  Early stopping at epoch {epoch+1} (patience={args.patience}).')
            break

    return best_val_metrics, best_score


# ---------------------------------------------------------------------------
# Optuna inner loop
# ---------------------------------------------------------------------------

def objective(
    trial: optuna.Trial,
    args: argparse.Namespace,
    outer_train_partitions: List[int],
    outer_fold: int,
    optuna_epochs: int,
) -> float:
    '''4-fold inner CV objective for Optuna hyperparameter search.

    Search space is calibrated for ESM3 (1536-dim) + propeptide-only CRF:
      - num_filters / hidden_size start at 32 (not 16) because ESM3 is more
        expressive and the model needs capacity to exploit it.
      - batch_size upper bound is 64 (not 128) to stay within GPU VRAM when
        sequences are padded to their longest member (1536 × L × batch).
      - weight_decay is searched to regularize against the richer ESM3 features.
    '''
    # Search space narrowed for ESM3 (1536-dim) + small Optuna budget (5 trials).
    #
    # Architecture data-flow: (batch,1536,L) → Conv1(1536→n_filters) → biLSTM
    # → Conv2(hidden*2→n_filters*2) → Linear(n_filters*2→2) → 51-state CRF
    #
    # Fixed (not searched):
    #   batch_size=64  — largest stable batch; fewer iterations per epoch = faster
    #   kernel_size=3  — short local context is what matters for cleavage sites
    #   weight_decay=1e-4 — light L2, not critical with only ~6k sequences
    #
    # Searched (most impact on ESM3 performance):
    #   lr           — most sensitive; LR scheduler handles decay automatically
    #   num_filters  — controls 1536→n_filters compression bottleneck (64 = 24:1,
    #                  128 = 12:1); must be ≥64 for ESM3
    #   hidden_size  — LSTM state size; must be ≥ n_filters/2 to avoid underfitting
    #   dropout      — single dropout rate applied at both input and conv layers
    args.batch_size   = 64
    args.kernel_size  = 3
    args.weight_decay = 1e-4

    args.lr          = trial.suggest_float('lr', 5e-5, 3e-4, log=True)
    args.num_filters = trial.suggest_categorical('num_filters', [64, 128])
    args.hidden_size = trial.suggest_categorical('hidden_size', [64, 128])
    args.dropout     = trial.suggest_float('dropout', 0.1, 0.3)
    args.conv_dropout = args.dropout  # tie the two dropout rates

    inner_scores = []
    orig_epochs = args.epochs
    args.epochs = optuna_epochs  # use shorter budget for HP search

    try:
        for inner_i, inner_val in enumerate(outer_train_partitions):
            inner_train = [p for p in outer_train_partitions if p != inner_val]
            train_loader, valid_loader, _ = get_dataloaders(
                args, inner_train, [inner_val], [inner_val],
            )
            model = get_model(args).to(device)
            _flatten_lstm_if_present(model)
            optimizer = Adam(model.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay)
            ckpt = os.path.join(
                args.out_dir,
                f'tmp_outer{outer_fold}_trial{trial.number}_inner{inner_i}.pt',
            )
            writer = SummaryWriter(
                os.path.join(args.out_dir, f'trial{trial.number}_outer{outer_fold}_inner{inner_i}')
            )

            _, score = run_training_for_params(
                args, model, train_loader, valid_loader, optimizer, writer, ckpt,
            )
            inner_scores.append(score)
            writer.close()

            if os.path.exists(ckpt):
                os.remove(ckpt)
    finally:
        args.epochs = orig_epochs  # always restore so retraining uses full epoch budget

    return float(np.mean(inner_scores))


# ---------------------------------------------------------------------------
# Outer 5-fold nested CV  (main entry point)
# ---------------------------------------------------------------------------

def train_nested_cv(args: argparse.Namespace) -> Dict:
    '''
    Full 5-fold nested cross-validation as described in DeepPeptide (2023).

    Outer loop  : 5 folds, each with 1 test partition and 4 training partitions.
    Inner loop  : Optuna optimizes hyperparameters via 4-fold CV on the outer
                  training partitions (using --optuna_epochs epochs).
    Models saved: all 4 inner-fold models per outer fold → 5×4 = 20 models for
                  ensemble inference.

    Use --outer_fold N (0-4) to run a single fold in parallel with other processes.
    '''
    if args.num_cpu_threads:
        torch.set_num_threads(args.num_cpu_threads)
        print(f'PyTorch CPU threads: {args.num_cpu_threads}', flush=True)

    optuna_epochs = args.optuna_epochs if args.optuna_epochs else args.epochs

    all_partitions = list(range(5))
    all_outer_results = []

    # Allow running a single outer fold (for parallel execution across processes).
    fold_range = [args.outer_fold] if args.outer_fold is not None else range(5)

    for outer_fold in fold_range:
        print(f'\n=== Outer fold {outer_fold} ===')
        test_partition = [outer_fold]
        outer_train_partitions = [p for p in all_partitions if p != outer_fold]

        # ---- Inner loop: Optuna ----
        study = optuna.create_study(direction='maximize')

        def _objective_with_log(trial):
            print(f'\n--- Outer {outer_fold} | Trial {trial.number+1}/{args.n_trials} ---', flush=True)
            score = objective(trial, args, outer_train_partitions, outer_fold, optuna_epochs)
            print(f'    Trial {trial.number+1} score: {score:.4f}', flush=True)
            return score

        study.optimize(
            _objective_with_log,
            n_trials=args.n_trials,
        )
        best_params = study.best_params
        print(f'Best hyperparameters (outer fold {outer_fold}):', best_params)
        for k, v in best_params.items():
            setattr(args, k, v)
        args.conv_dropout = args.dropout  # conv_dropout is tied to dropout but not in best_params

        json.dump(
            best_params,
            open(os.path.join(args.out_dir, f'best_params_outer{outer_fold}.json'), 'w'),
            indent=2,
        )

        # ---- Retrain 4 inner-fold models with best hyperparameters ----
        outer_fold_test_metrics = []

        for inner_i, inner_val in enumerate(outer_train_partitions):
            inner_train = [p for p in outer_train_partitions if p != inner_val]

            train_loader, valid_loader, test_loader = get_dataloaders(
                args, inner_train, [inner_val], test_partition,
            )

            model = get_model(args).to(device)
            _flatten_lstm_if_present(model)
            optimizer = Adam(model.parameters(), lr=args.lr,
                             weight_decay=getattr(args, 'weight_decay', 0.0))

            ckpt_path = os.path.join(
                args.out_dir, f'model_outer{outer_fold}_inner{inner_i}.pt'
            )
            writer = SummaryWriter(
                os.path.join(args.out_dir, f'outer{outer_fold}_inner{inner_i}')
            )

            _, _ = run_training_for_params(
                args, model, train_loader, valid_loader, optimizer, writer, ckpt_path,
            )
            writer.close()

            # Evaluate on outer test partition
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            _, test_probs, test_preds, _, test_labels = run_dataloader(
                test_loader, model, optimizer, writer, do_train=False,
            )
            test_metrics = compute_all_metrics(
                test_probs, test_preds, test_labels,
                test_loader.dataset.names, test_loader.dataset.data,
                windows=[3],
            )[0]
            test_metrics['outer_fold'] = outer_fold
            test_metrics['inner_fold'] = inner_i
            outer_fold_test_metrics.append(test_metrics)

            pickle.dump(
                (test_probs, test_preds, test_labels, test_loader.dataset.names),
                open(os.path.join(
                    args.out_dir, f'test_outputs_outer{outer_fold}_inner{inner_i}.pickle'
                ), 'wb'),
            )
            print(
                f'  inner fold {inner_i}: test F1 propeptides = '
                f'{test_metrics["f1 propeptides"]:.4f}'
            )

        all_outer_results.extend(outer_fold_test_metrics)

    # ---- Aggregate results (only when all 5 folds ran in this process) ----
    if args.outer_fold is None:
        f1_scores = [m['f1 propeptides'] for m in all_outer_results]
        summary = {
            'mean_f1_propeptides': float(np.mean(f1_scores)),
            'std_f1_propeptides':  float(np.std(f1_scores)),
            'n_models': len(all_outer_results),
            'per_model': all_outer_results,
        }
        json.dump(summary, open(os.path.join(args.out_dir, 'nested_cv_summary.json'), 'w'), indent=2)
        print(
            f'\nNested CV complete. '
            f'Mean propeptide F1 = {summary["mean_f1_propeptides"]:.4f} '
            f'± {summary["std_f1_propeptides"]:.4f}'
        )
    else:
        f1_scores = [m['f1 propeptides'] for m in all_outer_results]
        print(
            f'\nOuter fold {args.outer_fold} complete. '
            f'Mean test F1 = {np.mean(f1_scores):.4f}. '
            f'Run evaluation/measure_performance.py once all 5 folds finish.'
        )
        summary = {'outer_fold': args.outer_fold, 'per_model': all_outer_results}
    return summary


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument('--embeddings_dir', type=str, default='/data3/fegt_data/embeddings/')
    p.add_argument('--data_file', '-df', type=str,
                   default='data/uniprot_12052022_cv_5_50/labeled_sequences.csv')
    p.add_argument('--partitioning_file', '-pf', type=str,
                   default='data/uniprot_12052022_cv_5_50/graphpart_assignments.csv')
    p.add_argument('--embedding', '-em', type=str, default='precomputed')
    p.add_argument('--embedding_dim', '-ed', type=int, default=1536)

    p.add_argument('--model', '-m', type=str, default='lstmcnncrf')

    p.add_argument('--out_dir', '-od', type=str, default='train_run')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--patience', type=int, default=10,
                   help='Early stopping patience (epochs without improvement).')
    p.add_argument('--batch_size', '-bs', type=int, default=64)
    p.add_argument('--n_trials', type=int, default=50,
                   help='Number of Optuna trials per outer fold.')
    p.add_argument('--num_workers', type=int, default=0,
                   help='DataLoader workers. 0 (default) keeps embeddings cached in main process.')
    p.add_argument('--outer_fold', type=int, default=None, choices=[0, 1, 2, 3, 4],
                   help='Run only this outer fold. Omit to run all 5 sequentially.')
    p.add_argument('--num_cpu_threads', type=int, default=None,
                   help='torch.set_num_threads(). Set to ~(total_cores/5) when running folds in parallel.')
    p.add_argument('--optuna_epochs', type=int, default=None,
                   help='Max epochs for Optuna HP search (default: same as --epochs). '
                        'Use 35 to cover the typical phase-transition zone (~22-30 epochs) '
                        'while saving ~30%% vs full 50 epochs. Retraining always uses --epochs.')

    p.add_argument('--use_focal', default=True, action=argparse.BooleanOptionalAction,
                   help='Add focal loss on emissions to combat class imbalance (--no-use-focal to disable).')

    # These are the starting defaults; Optuna will override them during search.
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--conv_dropout', type=float, default=0.1)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--kernel_size', type=int, default=3)
    p.add_argument('--num_filters', type=int, default=64)
    p.add_argument('--hidden_size', type=int, default=64)

    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    # Use fold-specific filename when running in parallel to avoid race conditions.
    cfg_name = f'config_outer{args.outer_fold}.json' if args.outer_fold is not None else 'config.json'
    json.dump(vars(args), open(os.path.join(args.out_dir, cfg_name), 'w'), indent=3)
    return args


if __name__ == '__main__':
    train_nested_cv(parse_arguments())
