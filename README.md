# DeepPeptide (ESM3, multimodal) — propeptide cleavage prediction
Propeptide-only CRF over ESM3 embeddings (`esm3_sm_open_v1`, 1536-dim), adapted from
[DeepPeptide](https://github.com/fteufel/DeepPeptide) (Teufel et al., *Bioinformatics* 2023).

`esm3-propeptide` with the structure conditioning turned into a powered experiment
rather than a single pair of runs.

ESM3 is natively multimodal — sequence, backbone coordinates, VQ-VAE structure
tokens, SASA and ss8 are all input tracks. The reported ESM3 numbers used the
sequence track alone. The existing structure result is one run against one run
(−0.0568 F1, entirely in recall) against a ±0.0409 replicate band, which is why
RESULTS.md files it under *suggestive, not established*.

**Nothing new has been run on this branch.** It adds the per-pathway controls, the
negative control, the dual-tolerance reporting and the run plan. The design, the
power calculation and the reading order are in [NEXT_STEPS.md](NEXT_STEPS.md).

### Structural arms

| arm | flags | what it isolates |
|---|---|---|
| sequence-only | `--no_structure` | the same code path with tracks masked |
| all tracks | *(none)* | the treatment |
| tokens only | `--no_coords` | the VQ-VAE pathway |
| geometry only | `--no_struct_tokens` | Geometric Attention |
| **scrambled control** | `--scramble_structure` | tracks present, residue axis permuted, no true fold |

The scrambled control is the one that matters. ESM3 keeps 1536 dims whatever it
is fed, so the structure contrast is not confounded by dimensionality — but it is
still confounded by "having a second track at all", and only the control
separates that from geometry.

**The architecture is unchanged, for every arm.** Structure enters through
ESM3's own conditioning rather than by widening the input, so all five arms
build a byte-identical model to `esm3-propeptide` — 51 CRF states, the same
constraint mask, 241,521 trainable parameters. `test_architecture.py` asserts it
against values measured on the parent branch:

```bash
python test_architecture.py
```

Passing `--max_peptide_len 50 --min_peptide_len 5` explicitly builds the same
model as passing nothing, which is what makes the grammar flags safe to have.

The consolidated ESM3 branch also keeps: sequence-only and structure-conditioned
extractors, LoRA fine-tuning, the Optuna nested-CV search, ensembling and the
analysis tooling.

> **⚠ ESM3 embeddings made before 2026-08-19 are mis-scaled by ~840× and every result
> from them is invalid.** They can be repaired without re-running ESM3, and `preflight.py`
> refuses to start on them. See [EMBEDDINGS.md](EMBEDDINGS.md).

### Also new here

- **Both boundary tolerances are scored.** Every run writes `f1 propeptides@1`
  and `f1 propeptides@3` to `test_metrics.json`. The unsuffixed
  `f1 propeptides` key still holds ±3 and model selection still uses ±3, so
  every number in [RESULTS.md](RESULTS.md) stays valid and comparable.
- **`valid_metrics.json` is written on the single-run path.** RESULTS.md notes
  that 68 of 84 finished runs have no auditable record of which epoch was
  selected. Fixed.
- **The CRF grammar is configurable** via `--min_peptide_len` /
  `--max_peptide_len`, defaulting to the published 5..50 window.
  [GRAMMAR.md](GRAMMAR.md) measures what that window covers in UniProt and what
  widening it would cost; `test_grammar.py` asserts the defaults reproduce the
  published grammar exactly.

### Before you run this
This code accompanies an MSc thesis. **If you intend to run it, please contact me first**
— apostolosfysekidis1@gmail.com. The trained weights and search outputs are not published
here and are available on request. MIT licensed, so this is a request, not a condition.

### Quick start
```bash
conda create -n deeppeptide python=3.10 -y && conda activate deeppeptide
pip install -r requirements.txt          # needs a CUDA torch build

bash run_optuna_gpu.sh --fold 0 \
    --embeddings_dir /path/to/embeddings/esm3_normed \
    --out_dir results/esm3_prop_optuna_normed     # --out_dir MUST be new

python summarize_optuna.py --out_dir results/esm3_prop_optuna_normed
```
`run_optuna_gpu.sh` runs `preflight.py` itself and aborts if it fails.

### Documentation
- [TRAINING.md](TRAINING.md) — what you need, choosing a protocol, options, output
- [EMBEDDINGS.md](EMBEDDINGS.md) — the embedding scaling bug and how to repair it
- [OPTUNA_GPU.md](OPTUNA_GPU.md) — the search space, where it comes from, and its cost
- [EXPERIMENT.md](EXPERIMENT.md) — the protocol governing every arm of the comparison
- [RESULTS.md](RESULTS.md) — measured results
- [CHANGELOG.md](CHANGELOG.md) — what differs from the original DeepPeptide
- [predictor/README.md](predictor/README.md) — inference with the pretrained model
