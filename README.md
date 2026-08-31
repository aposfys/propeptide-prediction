# DeepPeptide (ESM3) — propeptide cleavage prediction
Propeptide-only CRF over ESM3 embeddings (`esm3_sm_open_v1`, 1536-dim), adapted from
[DeepPeptide](https://github.com/fteufel/DeepPeptide) (Teufel et al., *Bioinformatics* 2023).

The consolidated ESM3 branch: sequence-only and structure-conditioned extractors, LoRA
fine-tuning, the Optuna nested-CV search, ensembling and the analysis tooling.

> **⚠ ESM3 embeddings made before 2026-08-19 are mis-scaled by ~840× and every result
> from them is invalid.** They can be repaired without re-running ESM3, and `preflight.py`
> refuses to start on them. See [EMBEDDINGS.md](EMBEDDINGS.md).

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
