# Changelog — DeepPeptide ESM3 (propeptide-only)

Adaptation of DeepPeptide (Teufel et al., *Bioinformatics* 2023) to use
ESM3 (`esm3_sm_open_v1`, 1536-dim) as the sequence encoder and to predict
**propeptide cleavage sites only**, removing all mature-peptide logic.

---

## 1. Sequence Encoder: ESM1b/ESM2 → ESM3

| Aspect | Original | This version |
|---|---|---|
| Library | `fair-esm==1.0.2` | `esm>=3.0.0` |
| Model | `esm2_t33_650M_UR50D` (33 layers, 1280-dim) | `esm3_sm_open_v1` (1536-dim) |
| Load API | `pretrained.load_model_and_alphabet()` | `ESM3.from_pretrained()` |
| Input API | `batch_converter(seqs)` + `toks` | `ESMProtein(sequence=seq)` → `esm_model.encode()` |
| Token tensor | `encoded.sequence_tokens` | `encoded.sequence.unsqueeze(0)` |
| Output extraction | `out["representations"][33]` with `repr_layers=33` | `out.embeddings[0, 1:-1]` (strips CLS/EOS) |
| Long-sequence handling | Sliding 1022-token chunks with 300-token overlap | Not needed (ESM3 handles full sequences) |
| FASTA loading | `FastaBatchedDataset.from_file()` (crashed on duplicate labels) | Custom `_read_fasta()` with MD5 deduplication |
| Embedding dim default | 1280 | 1536 |

**Files changed:** `src/utils/make_embeddings.py`, `src/models/lstm_cnn.py`, `src/models/crf_models.py`

---

## 2. CRF Architecture: Two-branch → Propeptide-only

The original model jointly predicted mature peptides and propeptides via a
two-branch CRF. This version predicts propeptides only.

| Aspect | Original | This version |
|---|---|---|
| Branches | Mature peptide (states 1–50) + propeptide (states 51–100) + background (0) | Propeptide only (states 1–50) + background (0) |
| Total CRF states | 101 (two-branch) or 51 (peptide-only) | **51** always |
| `num_labels` | 3 (two-branch) or 2 (single-branch) | **2** always |
| `label_type` argument | `'multistate_with_propeptides'` controlled branching | Removed — propeptide labels always |
| `_repeat_emissions` | Handled both 2- and 3-label cases | 2-label case only |
| Skip connections | From state 3 to states 5–50 (peptide) + mirrored for propeptide branch | From state 3 to states 5–49 (propeptide only) |

**Files changed:** `src/models/crf_models.py`, `src/train_loop_crf.py`

---

## 3. Training Protocol

### 3a. Nested cross-validation

| Aspect | Original | This version |
|---|---|---|
| Entry point | `train(parse_arguments())` — single train/val/test split | `train_nested_cv(parse_arguments())` — 5-fold nested CV |
| Outer loop | None | 5 outer folds, one held-out test partition each |
| Inner loop | None (HPs hardcoded or manually tuned) | Optuna 4-fold CV over outer training partitions |
| HP trials | None | `--n_trials` per outer fold (default 50, recommended 5) |
| Models saved | One checkpoint | 5 outer × 4 inner = **20 checkpoints** for ensemble inference |
| Optuna objective | — | Mean validation F1 (propeptide) across 4 inner folds |

### 3b. Hyperparameter search space (Optuna)

| Parameter | Original default | Searched range |
|---|---|---|
| `lr` | 1e-4 (fixed) | log-uniform [5e-5, 3e-4] |
| `num_filters` | 32 | categorical {64, 128} |
| `hidden_size` | 64 | categorical {64, 128} |
| `dropout` | 0.1 | uniform [0.1, 0.3] |
| `batch_size` | 100 | fixed at 64 |
| `kernel_size` | 3 | fixed at 3 |
| `weight_decay` | — | fixed at 1e-4 |

### 3c. LR scheduler

| Aspect | Original | This version |
|---|---|---|
| Scheduler | None (constant LR) | Linear warmup + cosine decay |
| Warmup | — | `max(3, int(0.05 × epochs))` epochs |
| Decay | — | Cosine to floor of 1e-6 |
| Implementation | — | `torch.optim.lr_scheduler.LambdaLR` |

### 3d. Other training changes

| Aspect | Original | This version |
|---|---|---|
| Distributed training | FSDP (`fairscale`) wrapper | Removed entirely — single-process |
| Early stopping | Not implemented | Patience on validation F1 (`--patience`, default 10) |
| Epoch default | 30 | 50 |
| Train shuffle | `False` | `True` |
| Gradient clipping | 0.25 (already present) | 0.25 (retained) |

**Files changed:** `src/train_loop_crf.py`

---

## 4. Model Initialisation

| Aspect | Original | This version |
|---|---|---|
| Conv1d / Conv2d | PyTorch default (Kaiming uniform) | Explicit `kaiming_uniform_(nonlinearity='relu')` |
| biLSTM weights | PyTorch default (uniform) | Explicit `xavier_uniform_` per weight matrix |
| All biases | PyTorch default | Explicit zeros |
| Emission Linear | PyTorch default | Explicit `xavier_uniform_` |

`LSTMCNNCRF._init_weights()` was added to enforce these at construction time.

**Files changed:** `src/models/crf_models.py`

---

## 5. Loss Function

| Aspect | Original | This version |
|---|---|---|
| CRF NLL | Yes | Yes |
| Auxiliary loss | None | Focal loss on raw emissions (`weight = 0.1`) |
| Focal computation | — | `log_softmax` over both logits (consistent with CRF) |
| Class weighting | — | Inverse-class-frequency alpha per batch |
| Focal gamma | — | 2.0 |
| Toggle | — | `--use_focal` / `--no-use-focal` (default: on) |

The focal loss was introduced to combat class imbalance (propeptide residues
are a small fraction of each sequence). Using `log_softmax` over both logits
instead of `sigmoid(logit_1)` keeps the computation consistent with how the
CRF reads emissions via `_repeat_emissions`.

**Files changed:** `src/models/crf_models.py`

---

## 6. Dropout Layer Fix

| Aspect | Original | This version |
|---|---|---|
| Dropout class | `nn.Dropout2d` | `nn.Dropout1d` |
| Input shape | 3D `(N, C, L)` — wrong for Dropout2d | 3D `(N, C, L)` — correct for Dropout1d |
| Effect | Identical (channel-wise dropout), but UserWarning every batch | Identical, no warning |

`nn.Dropout2d` expects 4D input `(N, C, H, W)`. Using it on 3D tensors
triggered a PyTorch UserWarning that would become a behaviour change in
future releases. `nn.Dropout1d` is the correct module for `(N, C, L)`.

**Files changed:** `src/models/lstm_cnn.py`

---

## 7. Evaluation Script

| Aspect | Original | This version |
|---|---|---|
| Checkpoint discovery | Hardcoded list of 20 ESM2 model paths | Auto-discovers `test_outputs_outer*_inner*.pickle` from `--out_dir` |
| Data file | Hardcoded path | `--data_file` argument |
| CRF state for propeptide | `START=51, END=100` (wrong — two-branch states) | `START=1, END=50` (correct for propeptide-only) |
| Division-by-zero guard | None (crashed when no true positives) | `if (tp + fp) > 0 else 0.0` |
| Output | Console only | Saves `metrics_per_model.csv` and `metrics_aggregated.csv` |
| Tolerances reported | Fixed | P/R/F1 at tolerances 0–3 |

**Files changed:** `evaluation/measure_performance.py`

---

## 8. CPU Parallelism (new)

Not present in the original. Added to run the 5-fold nested CV on multi-core
CPU nodes without GPU access.

| Feature | Description |
|---|---|
| `--outer_fold N` | Run only fold N (0–4); enables 5 parallel processes |
| `--num_cpu_threads N` | Sets `torch.set_num_threads(N)`; propagated via OMP/MKL env vars |
| `--optuna_epochs N` | Shorter epoch budget for HP search (recommended: 35); retraining always uses `--epochs` |
| `run_parallel.sh` | Launches 5 outer folds via `taskset -c START-END`; waits for all and reports failures |
| Embedding cache | `_EMBEDDING_CACHE` dict in `dataset.py`: each `.pt` file loaded once per process, served from RAM every subsequent epoch; first epoch is still disk-bound |
| `num_workers` default | Changed from 2 to 0 (keeps cache in main process, not forked workers) |
| Config race condition | Each fold writes `config_outer{N}.json` instead of a shared `config.json` |

**Files added:** `run_parallel.sh`  
**Files changed:** `src/train_loop_crf.py`, `src/utils/dataset.py`

---

## 9. Bug Fixes

| Bug | File | Fix |
|---|---|---|
| `ImportError: cannot import name 'train'` | `run.py` | Changed import to `train_nested_cv` |
| `AssertionError: Found duplicate sequence labels` in `FastaBatchedDataset.from_file()` | `make_embeddings.py` | Replaced with custom `_read_fasta()` that deduplicates by MD5 hash |
| `AttributeError: 'ESMProteinTensor' has no attribute 'sequence_tokens'` | `make_embeddings.py`, `lstm_cnn.py` | Changed to `encoded.sequence` |
| `IndexError: tuple index out of range` in ESM3 forward | `make_embeddings.py`, `lstm_cnn.py` | Added `.unsqueeze(0)` for batch dimension |
| `conv_dropout` not restored after Optuna trial | `train_loop_crf.py` | Added `args.conv_dropout = args.dropout` after restoring `best_params`; epoch override wrapped in `try/finally` |
| Wrong propeptide CRF states in evaluation | `evaluation/measure_performance.py` | Changed `START=51,END=100` → `START=1,END=50` |
| Division-by-zero in precision | `evaluation/measure_performance.py` | Added guard when `tp + fp == 0` |
| `Dropout2d` UserWarning on 3D input | `src/models/lstm_cnn.py` | Replaced with `Dropout1d` |

---

## 10. Dependencies

| Package | Original | This version |
|---|---|---|
| ESM | `fair-esm==1.0.2` | `esm>=3.0.0` |
| PyTorch | `torch==1.11.0` | `torch>=2.0.0` |
| Optuna | `optuna==2.10.0` | `optuna>=3.0.0` |
| fairscale | Required (FSDP) | **Removed** |

---

## 11. Files Unchanged

| File | Reason |
|---|---|
| `baselines/` | Standalone PeptideLocator wrapper; not part of the ESM3 pipeline |
| `src/models/multi_tag_crf.py` | CRF kernel is model-agnostic; no changes needed |
| `src/utils/crf_label_utils.py` | Label encoding logic is encoder-agnostic |
| `data/` | Same dataset (UniProt 2022-05-12, 5-fold Graph-Part split) |
