'''
CRF train loop — propeptide cleavage site prediction via ESM3 + LSTM-CNN-CRF.

Training protocol (DeepPeptide, Bioinformatics 2023):
  - 50 epochs, constant Adam LR, best-on-validation checkpointing (metric:
    propeptide F1). No scheduler and no early stopping, matching upstream.
    --search_patience applies inside the Optuna search only, as a declared
    budget measure; reported models train on the full budget.
  - 5-fold nested cross-validation: Optuna inner loop (4-fold) finds best
    hyperparameters; the 4 inner-fold models per outer fold are saved and
    used as a 5×4=20-model ensemble at inference.
  - ESM3 weights are frozen; only the prediction head is trained.
  - Loss: negative log-likelihood of the CRF (Viterbi / forward-backward).
'''
import json
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import optuna
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .models import LSTMCNNCRF, SimpleLSTMCNNCRF, SelfAttentionCRF
from .utils.dataset import PrecomputedCSVForOverlapCRFDataset
from .utils.manuscript_metrics import compute_all_metrics

import argparse

# This branch is GPU-only. Device selection still falls back so that the module
# can be imported, --help works, and preflight can run on a CPU login node — but
# any actual training run is gated by require_cuda() below.
device = torch.device('cuda') if torch.cuda.is_available() else (torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu'))
global_step = 0


def require_cuda(args: argparse.Namespace) -> None:
    '''Abort unless a CUDA GPU is present.

    The hyperparameter search on this branch is a GPU workload: one trial is 4
    inner-fold trainings, and a full search is hundreds of them. On CPU the same
    run takes orders of magnitude longer, so silently falling back produces a job
    that looks alive for days and never finishes. Failing here, immediately, is
    the useful behaviour.

    --allow_cpu exists only to smoke-test plumbing on a login node. It is not for
    producing results.
    '''
    if device.type == 'cuda':
        print(f'Device: cuda — {torch.cuda.get_device_name(0)} '
              f'({torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB)',
              flush=True)
        return

    if getattr(args, 'allow_cpu', False):
        print(
            f'WARNING: --allow_cpu set, running on {device.type}. This is for '
            'smoke-testing plumbing only — do NOT report results from this run.',
            flush=True,
        )
        return

    raise SystemExit(
        f'ERROR: no CUDA GPU available (torch sees "{device.type}").\n'
        '\n'
        'This branch is GPU-only — a full search is hundreds of trainings and is\n'
        'not viable on CPU. Run it on a GPU node.\n'
        '\n'
        'If a GPU should be visible here, check:\n'
        '  python -c "import torch; print(torch.__version__, torch.version.cuda)"\n'
        '  nvidia-smi\n'
        'A CPU-only torch build reports cuda=None and needs reinstalling.\n'
        '\n'
        'To smoke-test the plumbing on a CPU node anyway, pass --allow_cpu\n'
        '(results from such a run are not usable).'
    )


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
    # pin_memory speeds up the host->GPU copy of the padded embedding batches,
    # which are the largest tensors moved each step (1536 x L_max x batch).
    pin = device.type == 'cuda'
    # INVARIANT: the eval loaders must stay shuffle=False. compute_all_metrics is
    # handed predictions in *loader* order but names/data in *dataset* order, and
    # pairs them positionally. Equal lengths make any permutation invisible — no
    # exception, just silently mismatched metrics. Introducing shuffling, length
    # bucketing, or a custom sampler on valid/test requires inverting the
    # permutation first and asserting the names line up.
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        collate_fn=train_set.collate_fn, num_workers=nw, pin_memory=pin,
    )
    valid_loader = DataLoader(
        valid_set, batch_size=args.batch_size, shuffle=False,
        collate_fn=valid_set.collate_fn, num_workers=max(0, nw - 1), pin_memory=pin,
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        collate_fn=test_set.collate_fn, num_workers=max(0, nw - 1), pin_memory=pin,
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
    collect_probs: bool = True,
) -> Tuple[float, List[np.ndarray], List[List[int]], List[np.ndarray], List[np.ndarray]]:
    '''Run one epoch; collect per-sequence Viterbi paths, marginals, and labels.

    With do_train=True only the mean loss is populated — the four collection
    lists come back empty, because every caller discards them there and
    producing them costs a Viterbi decode plus a host copy per batch.

    collect_probs=False additionally drops the marginals and labels in eval.
    compute_all_metrics scores the Viterbi paths alone, so the per-epoch
    validation pass needs neither; only the final test pass does, to pickle them.
    '''
    global global_step

    true_peptides = []
    labels = []
    probs = []
    preds = []
    epoch_loss = []

    model.train() if do_train else model.eval()

    for batch in loader:
        embeddings, mask, label, propeptides = batch
        # non_blocking lets the copy overlap with compute. It is only effective
        # because the loaders set pin_memory=True on CUDA, and it matters here
        # because the padded embedding batch is the largest tensor moved each
        # step -- 1536 x L_max x batch floats, over 1 GB at batch 100 / L 2000.
        # Same values either way; PyTorch inserts the sync before first use.
        nb = device.type == 'cuda'
        embeddings = embeddings.to(device, non_blocking=nb)
        mask = mask.to(device, non_blocking=nb)
        label = label.long().to(device, non_blocking=nb)

        if do_train:
            model.zero_grad()
            pos_probs, pos_preds, loss = model(
                embeddings, mask, label, skip_marginals=True, use_focal=use_focal,
                skip_decode=True,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            loss_value = loss.item()
            writer.add_scalar('Train/loss', loss_value, global_step=global_step)
            global_step += 1
        else:
            with torch.no_grad():
                pos_probs, pos_preds, loss = model(
                    embeddings, mask, label, skip_marginals=True, use_focal=False,
                )
            loss_value = loss.item()

        # Only the loss is read back during training — callers discard the other
        # four return values — so the per-sample host copies are skipped there.
        # In eval, each tensor crosses to the host once per batch rather than
        # once per sequence; the per-sample split then happens on CPU.
        if not do_train:
            true_peptides.extend(propeptides)
            preds.extend(pos_preds)
            if collect_probs:
                probs_np = pos_probs.detach().cpu().numpy()
                labels_np = label.detach().cpu().numpy()
                probs.extend([probs_np[i] for i in range(probs_np.shape[0])])
                labels.extend([labels_np[i] for i in range(labels_np.shape[0])])
        epoch_loss.append(loss_value)

    return sum(epoch_loss) / len(epoch_loss), probs, preds, true_peptides, labels


def run_training_for_params(
    args: argparse.Namespace,
    model: torch.nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    writer: SummaryWriter,
    checkpoint_path: str,
    use_focal: bool = False,
) -> Tuple[Dict, float]:
    '''Train for args.epochs with a constant LR and best-on-validation checkpointing.

    Constant Adam LR, no scheduler, matching the original DeepPeptide loop and the
    esm2-propeptide model.

    args.patience == 0 disables the early-stopping break, which is upstream's
    behaviour and the default for reported runs. A patience break can only ever
    return an equal or worse model than the full budget: best-checkpoint
    selection already is early stopping, and the break merely stops paying for
    epochs that might still have improved — a plateau longer than --patience
    followed by a late rise is unrecoverable.

    The break is kept behind the flag for the Optuna search, which trains 4
    models per trial (--search_patience). That is a declared budget measure for
    ranking configurations and never touches a reported model.

    `use_focal` adds the auxiliary focal term on the emissions. It defaults to
    False, which is the faithful setting and the one the published ESM3/ESM-2
    propeptide numbers were produced with.
    '''
    global global_step
    global_step = 0

    best_score = -1.0
    best_val_metrics = None
    patience_counter = 0

    for epoch in range(args.epochs):
        train_loss, _, _, _, _ = run_dataloader(
            train_loader, model, optimizer, writer, do_train=True,
            use_focal=use_focal,
        )

        _, valid_probs, valid_preds, _, valid_labels = run_dataloader(
            valid_loader, model, optimizer, writer, do_train=False,
            collect_probs=False,
        )
        valid_metrics = compute_all_metrics(
            valid_probs, valid_preds, valid_labels,
            valid_loader.dataset.names, valid_loader.dataset.data,
            windows=[3],
        )[0]

        score = valid_metrics['f1 propeptides']
        writer.add_scalar('Valid/f1_propeptides', score, global_step=epoch)

        improved = score > best_score
        if improved:
            best_score = score
            best_val_metrics = {**valid_metrics, 'epoch': epoch}
            torch.save(model.state_dict(), checkpoint_path)
            patience_counter = 0
        else:
            patience_counter += 1

        marker = '*' if improved else ' '
        patience_str = f'{patience_counter}/{args.patience}' if args.patience > 0 else 'off'
        print(
            f'  {marker} epoch {epoch+1:3d}  loss={train_loss:.4f}  '
            f'val_f1={score:.4f}  best={best_score:.4f}  '
            f'patience={patience_str}',
            flush=True,
        )

        if args.patience > 0 and patience_counter >= args.patience:
            print(f'  Early stopping at epoch {epoch+1} (patience={args.patience}).')
            break

    return best_val_metrics, best_score


# ---------------------------------------------------------------------------
# Optuna inner loop
# ---------------------------------------------------------------------------

# Search-space presets, selected with --space.
#
# 'table_s1' (default) is the paper's own space (btad616 Table S1), the one the
# ESM-1b / ESM-2 models were tuned over. Use it for any result you intend to
# compare against the published ESM-2 numbers: tuning both embedders over the
# same space is what makes "ESM3 vs ESM-2" a statement about the embedder rather
# than about who got the bigger search.
#
# 'wide' loosens every bound and additionally searches weight_decay. It answers a
# different question — "what is the best this architecture can do with ESM3,
# ignoring comparability" — and it is strictly exploratory:
#   * it is NOT comparable to the ESM-2 T4 result;
#   * it needs a substantially larger --n_trials to cover the bigger volume;
#     re-using the table_s1 trial budget here samples more thinly and usually
#     scores WORSE, not better.
# Upper bounds are still finite because the model has physical limits: lr much
# above 5e-2 diverges the CRF NLL, and batch/hidden size are bounded by VRAM and
# wall-clock rather than by taste.
SEARCH_SPACES = {
    'table_s1': dict(
        lr=(1e-4, 1e-2), batch_size=(10, 100, 10), dropout=(0.0, 0.7),
        conv_dropout=(0.0, 0.7), kernel_size=(1, 5, 2),
        num_filters=(40, 128, 8), hidden_size=(16, 192, 16),
        weight_decay=None,           # Table S1 omits it; upstream leaves Adam at 0
    ),
    'wide': dict(
        lr=(1e-5, 5e-2), batch_size=(8, 128, 8), dropout=(0.0, 0.8),
        conv_dropout=(0.0, 0.8), kernel_size=(1, 9, 2),
        num_filters=(16, 256, 16), hidden_size=(16, 384, 16),
        weight_decay=(1e-8, 1e-2),   # searched log-uniform
    ),
}


def objective(
    trial: optuna.Trial,
    args: argparse.Namespace,
    outer_train_partitions: List[int],
    outer_fold: int,
    optuna_epochs: int,
) -> float:
    '''4-fold inner CV objective for Optuna hyperparameter search.

    Architecture data-flow: (batch,1536,L) → Conv1(1536→n_filters) → biLSTM
    → Conv2(hidden*2→n_filters*2) → Linear(n_filters*2→2) → 51-state CRF

    Provenance of the ranges. Upstream DeepPeptide (fteufel/DeepPeptide) ships no
    hyperparameter-search code — only the training loop with fixed CLI values — so
    the search space is taken from the paper instead: DeepPeptide (Bioinformatics
    2023, btad616) **Table S1**, the space used for the ESM-1b and ESM-2 models,
    reproduced here parameter-for-parameter. `weight_decay` is deliberately NOT
    searched, because Table S1 does not include it and upstream never exposes it
    (Adam runs at its default 0).

    Table S1                     lower    upper    distribution
      Learning rate              0.0001   0.01     log-uniform
      Batch size                 10       100      step 10
      Embedding dropout          0        0.7      uniform
      Convolution dropout        0        0.7      uniform
      Kernel size                1        5        step 2
      CNN channels               40       128      step 8
      LSTM hidden size           16       192      step 16

    The space is explored exactly as the paper describes it: "explored in nested
    cross-validation (best set determined by average validation performance on 4
    inner fold models)" — i.e. the mean over the 4 inner folds below. Table S2's
    per-fold winners (T0..T4) all fall inside these ranges, including the T4 set
    that produced the ESM-2 propeptide F1 of 0.626 (lr 0.0055, batch 20, dropout
    0.6902/0.2672, kernel 5, channels 48, hidden 48).

    Why Table S1's space for a *different* model. This is an ESM3 (1536-dim),
    propeptide-only (2-label / 51-state) model, not the ESM-1b/ESM-2 (1280-dim)
    one Table S1 was written for — so reusing the space is a deliberate choice,
    not an assumption that the two share an optimum:
      1. Comparability. ESM-2 got Table S1's space; giving ESM3 a different one
         would confound the comparison, because an F1 gap could then come from
         the search rather than from the embedder. Same space = clean read.
      2. The embedder change does not touch the searched axes. 1280 -> 1536 only
         alters conv1's *input* channel count, which is fixed by --embedding_dim
         and never searched. Every searched parameter acts downstream of that
         projection, so its useful range is essentially unmoved.
      3. The ranges are broad, not ESM-2-specific: lr spans two orders of
         magnitude, both dropouts span 0-0.7, channels 40-128, hidden 16-192.
    If ESM3's optimum really does sit outside this box, the search says so by
    piling every fold's winner against a bound — summarize_optuna.py checks for
    exactly that on lr and prints a warning telling you to widen it. Widen it
    from evidence, not preemptively.

    Nothing about the per-epoch training logic changes — that stays upstream's
    (constant Adam LR, no scheduler, no focal term). Only which values get tried.

    Every tunable is a real `suggest_*` call, so `study.best_params` is a
    complete, self-sufficient record of the winning configuration.
    '''
    space = SEARCH_SPACES[args.space]

    # Learning rate: log-uniform. Constant LR, no scheduler, so this single value
    # is the entire schedule. (The old range capped at 3e-4 and could not reach
    # T4's 5.5e-3.)
    args.lr = trial.suggest_float('lr', *space['lr'], log=True)

    # Batch size: stepped integer.
    args.batch_size = trial.suggest_int('batch_size', *space['batch_size'][:2],
                                        step=space['batch_size'][2])

    # Embedding and convolution dropout, searched INDEPENDENTLY: Table S1 lists
    # them as two parameters, and T4 pairs a heavy 0.6902 embedding dropout with a
    # much lighter 0.2672 conv dropout — tying them to one value makes that
    # optimum unreachable.
    args.dropout      = trial.suggest_float('dropout', *space['dropout'])
    args.conv_dropout = trial.suggest_float('conv_dropout', *space['conv_dropout'])

    # Kernel size: odd values only. That is what Table S1 specifies (1..5 step 2)
    # and also what LSTMCNN requires: conv1 pads with filter_size // 2, preserving
    # sequence length only for odd sizes. An even kernel yields L+1 and trips the
    # `bi_out.size(1) == seq_lengths.max()` assert downstream.
    args.kernel_size = trial.suggest_int('kernel_size', *space['kernel_size'][:2],
                                         step=space['kernel_size'][2])

    # CNN channels: conv1's 1536 -> n_filters compression, the main bottleneck.
    args.num_filters = trial.suggest_int('num_filters', *space['num_filters'][:2],
                                         step=space['num_filters'][2])

    # biLSTM hidden state size.
    args.hidden_size = trial.suggest_int('hidden_size', *space['hidden_size'][:2],
                                         step=space['hidden_size'][2])

    # Only the 'wide' preset searches weight_decay; table_s1 leaves it at the CLI
    # default (0.0), matching upstream's untouched Adam.
    if space['weight_decay'] is not None:
        args.weight_decay = trial.suggest_float('weight_decay', *space['weight_decay'], log=True)

    inner_scores = []
    orig_epochs = args.epochs
    orig_patience = args.patience
    args.epochs = optuna_epochs  # use shorter budget for HP search
    # The search trains 4 models per trial, so it cannot afford upstream's full
    # budget. Early stopping here is a declared budget measure for ranking
    # configurations; the retrained, reported model uses args.patience.
    args.patience = args.search_patience

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

            try:
                _, score = run_training_for_params(
                    args, model, train_loader, valid_loader, optimizer, writer, ckpt,
                    use_focal=args.use_focal,
                )
            finally:
                writer.close()
                if os.path.exists(ckpt):
                    os.remove(ckpt)  # inner-loop checkpoints are throwaway

            inner_scores.append(score)

            # Prune on the running mean: a config that is clearly behind after
            # 1-2 of the 4 inner folds is abandoned instead of burning the rest.
            trial.report(float(np.mean(inner_scores)), step=inner_i)
            if trial.should_prune():
                print(
                    f'    pruned after inner fold {inner_i} '
                    f'(mean so far {np.mean(inner_scores):.4f})',
                    flush=True,
                )
                raise optuna.TrialPruned()
    finally:
        args.epochs = orig_epochs  # always restore so retraining uses full epoch budget
        args.patience = orig_patience

    return float(np.mean(inner_scores))


# ---------------------------------------------------------------------------
def train(args, train_partitions=[0,1,2], valid_partitions=[3], test_partitions=[4]):
    '''Simple single train/val/test split — used by run_fold.py for 5-fold CV.'''
    if getattr(args, 'num_cpu_threads', None):
        torch.set_num_threads(args.num_cpu_threads)

    # NOTE: use the module-level `device`; run_dataloader moves its batches with
    # that global, so a local re-detection here would silently diverge from it.
    require_cuda(args)

    train_loader, valid_loader, test_loader = get_dataloaders(
        args, train_partitions, valid_partitions, test_partitions)

    model = get_model(args).to(device)
    _flatten_lstm_if_present(model)
    optimizer = Adam(model.parameters(), lr=args.lr,
                     weight_decay=getattr(args, 'weight_decay', 0.0))
    writer = SummaryWriter(args.out_dir)
    checkpoint = os.path.join(args.out_dir, 'model.pt')

    best_val_metrics, _ = run_training_for_params(
        args, model, train_loader, valid_loader, optimizer, writer, checkpoint,
        use_focal=getattr(args, 'use_focal', False))

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    _, test_probs, test_preds, _, test_labels = run_dataloader(
        test_loader, model, optimizer, writer, do_train=False)

    from src.utils.manuscript_metrics import compute_all_metrics
    test_metrics = compute_all_metrics(
        test_probs, test_preds, test_labels,
        test_loader.dataset.names, test_loader.dataset.data, windows=[3])[0]

    pickle.dump((test_probs, test_preds, test_labels, test_loader.dataset.names),
                open(os.path.join(args.out_dir, 'test_outputs.pickle'), 'wb'))
    json.dump(test_metrics, open(os.path.join(args.out_dir, 'test_metrics.json'), 'w'), indent=2)
    return best_val_metrics, test_metrics


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
    # getattr, to match train() -- callers that build an args Namespace by hand
    # (test_smoke.py) do not necessarily set every CLI field, and a direct
    # attribute read turns that into an AttributeError deep inside the run.
    if getattr(args, 'num_cpu_threads', None):
        torch.set_num_threads(args.num_cpu_threads)
        print(f'PyTorch CPU threads: {args.num_cpu_threads}', flush=True)

    require_cuda(args)
    torch.manual_seed(args.seed)

    # Record the space in the log: 'wide' results must never be reported next to
    # the ESM-2 T4 number as though they were comparable.
    print(f'Search space: {args.space}  ({args.n_trials} trials per outer fold)', flush=True)
    for name, bounds in SEARCH_SPACES[args.space].items():
        print(f'    {name:14} {bounds if bounds is not None else "not searched"}', flush=True)
    if args.space == 'wide':
        print(
            'NOTE: --space wide is exploratory. It is NOT comparable to the ESM-2 T4\n'
            '      result (which was tuned over table_s1), and a bigger space needs a\n'
            '      bigger --n_trials — reusing the table_s1 budget samples it thinly\n'
            '      and typically scores worse.',
            flush=True,
        )

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
        # Seeded sampler so the search is reproducible; offset by outer_fold so the
        # five folds explore different trial sequences instead of the same one.
        # Pruning is opt-in (--prune): the default NopPruner runs every trial to
        # completion, which is the plain exhaustive search.
        #
        # The study is persisted to SQLite so a crash or preemption does not throw
        # away the trials already paid for — a full search is hundreds of trainings
        # over days. Re-running the same command resumes where it stopped.
        #
        # The search space is in the filename on purpose: trials are only
        # comparable within one space, and resuming a table_s1 study under --space
        # wide would silently mix two different searches into one best_params.
        storage_path = os.path.abspath(
            os.path.join(args.out_dir, f'optuna_{args.space}_outer{outer_fold}.db')
        )
        study = optuna.create_study(
            direction='maximize',
            study_name=f'{args.space}_outer{outer_fold}',
            storage=f'sqlite:///{storage_path}',
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=args.seed + outer_fold),
            pruner=(optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
                    if args.prune else optuna.pruners.NopPruner()),
        )

        # optimize(n_trials=N) runs N *more* trials, so on a resume the budget has
        # to be reduced by what the store already holds or the search overshoots.
        # Only finished trials count: a trial left RUNNING by a crash produced no
        # value, and FAILed ones are worth re-drawing.
        done = len([t for t in study.trials
                    if t.state in (optuna.trial.TrialState.COMPLETE,
                                   optuna.trial.TrialState.PRUNED)])
        remaining = max(0, args.n_trials - done)
        if done:
            print(
                f'Resuming study {study.study_name} from {storage_path}: '
                f'{done} trials already stored, running {remaining} more.',
                flush=True,
            )

        def _objective_with_log(trial):
            print(f'\n--- Outer {outer_fold} | Trial {trial.number+1}/{args.n_trials} ---', flush=True)
            score = objective(trial, args, outer_train_partitions, outer_fold, optuna_epochs)
            # trial.params is only populated once objective() has run its suggest_*
            # calls, so it is logged here rather than before.
            print(f'    Trial {trial.number+1} score: {score:.4f}  params: {trial.params}', flush=True)
            return score

        if remaining:
            study.optimize(
                _objective_with_log,
                n_trials=remaining,
            )
        best_params = study.best_params
        n_pruned = len([t for t in study.trials
                        if t.state == optuna.trial.TrialState.PRUNED])
        print(
            f'Outer fold {outer_fold}: {len(study.trials)} trials '
            f'({n_pruned} pruned). Best inner-CV F1 = {study.best_value:.4f}',
            flush=True,
        )
        print(f'Best hyperparameters (outer fold {outer_fold}):', best_params)
        # Every tunable is a real suggest_* call, so best_params is complete —
        # no manual patch-ups needed here.
        for k, v in best_params.items():
            setattr(args, k, v)

        json.dump(
            best_params,
            open(os.path.join(args.out_dir, f'best_params_outer{outer_fold}.json'), 'w'),
            indent=2,
        )
        # Full effective config, so the retrained models can be reproduced from
        # one file without having to know which defaults were in play.
        json.dump(
            {**vars(args), 'best_inner_cv_f1': study.best_value,
             'n_trials_run': len(study.trials), 'n_trials_pruned': n_pruned},
            open(os.path.join(args.out_dir, f'effective_config_outer{outer_fold}.json'), 'w'),
            indent=2,
        )
        # Per-trial history, for the supervisor to inspect the search itself.
        study.trials_dataframe().to_csv(
            os.path.join(args.out_dir, f'optuna_trials_outer{outer_fold}.csv'), index=False,
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
                use_focal=args.use_focal,
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
        summary = {
            'outer_fold': args.outer_fold,
            'mean_f1_propeptides': float(np.mean(f1_scores)),
            'std_f1_propeptides':  float(np.std(f1_scores)),
            'n_models': len(all_outer_results),
            'per_model': all_outer_results,
        }
        # Persist it: train_nested_cv's return value is discarded by __main__, so
        # without this file a per-fold run leaves its metrics only in the log.
        json.dump(
            summary,
            open(os.path.join(args.out_dir, f'fold_summary_outer{args.outer_fold}.json'), 'w'),
            indent=2,
        )
        # Say what actually happened. This used to end with "Run summarize_optuna.py
        # once all 5 folds finish", which on a deliberate single-fold run reads as
        # though the job stopped early with four folds outstanding. --outer_fold
        # runs exactly one fold by design; it is the paper's ablation protocol.
        print(
            f'\nOuter fold {args.outer_fold} complete. '
            f'Mean test F1 = {np.mean(f1_scores):.4f} '
            f'(written to fold_summary_outer{args.outer_fold}.json).\n'
            f'This fold is finished. --outer_fold runs one fold by design, so nothing\n'
            f'is outstanding unless you intend to run the other four separately.\n'
            f'Summarise with:  python summarize_optuna.py --out_dir {args.out_dir}'
        )
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
    p.add_argument('--patience', type=int, default=0,
                   help='Early stopping patience (epochs without improvement). '
                        '0 = disabled, run the full epoch budget and keep the '
                        'best-on-validation checkpoint. This is upstream '
                        'DeepPeptide behaviour and the default for reported runs.')
    p.add_argument('--search_patience', type=int, default=10,
                   help='Early stopping patience used inside the Optuna search '
                        'only (4 models per trial). Does not affect the '
                        'retrained, reported model. 0 = disabled.')
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

    p.add_argument('--use_focal', default=False, action=argparse.BooleanOptionalAction,
                   help='Add an auxiliary focal loss on the emissions to combat class '
                        'imbalance (--use_focal to enable, --no-use_focal to disable). '
                        'Default off here, which is the faithful loop. NOTE: the '
                        'esm3-propeptide branch defaults it ON, and the archived ESM3 '
                        'propeptide run (results/esm3_prop_default) was produced with '
                        'use_focal=true while the ESM-2 T4 baseline was not — so those '
                        'two are not like-for-like. Pass it explicitly when comparing '
                        'embedders.')
    p.add_argument('--space', type=str, default='table_s1', choices=sorted(SEARCH_SPACES),
                   help="Search space preset. 'table_s1' (default) is the paper's own "
                        'space (btad616 Table S1) — use it for anything compared against '
                        "the published ESM-2 numbers. 'wide' loosens every bound and also "
                        'searches weight_decay; exploratory only, not comparable to the '
                        'ESM-2 T4 result, and needs a larger --n_trials to be meaningful.')
    p.add_argument('--seed', type=int, default=42,
                   help='Seed for the Optuna TPE sampler and torch, so a search is reproducible.')
    p.add_argument('--prune', action='store_true',
                   help='Opt in to Optuna median pruning: abandon a trial once its '
                        'running inner-fold mean is clearly behind. Saves GPU time but '
                        'makes the search non-exhaustive. Off by default.')
    p.add_argument('--allow_cpu', action='store_true',
                   help='Override the GPU requirement. For smoke-testing plumbing on a '
                        'CPU node only — a real search is not viable on CPU, and results '
                        'from such a run must not be reported.')

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
