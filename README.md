# DeepPeptide (ESM3)
Predicting propeptide cleavage sites in protein sequences using ESM3.

Adapted from the original DeepPeptide (Teufel et al., Bioinformatics 2023) to use
ESM-3 (`esm3_sm_open_v1`, 1536-dim) as the sequence encoder and a propeptide-only
CRF (51 states: background + 50 propeptide positions).

### Training the model
1. Precompute ESM3 embeddings:
```
python -m src.utils.make_embeddings \
    data/protein_sequences.fasta \
    PATH/TO/EMBEDDINGS/
```

2. Train with 5-fold nested CV (Optuna inner loop, 5 trials):
```
python -m src.train_loop_crf \
    --embeddings_dir PATH/TO/EMBEDDINGS \
    --data_file data/labeled_sequences.csv \
    --partitioning_file data/graphpart_assignments.csv \
    --out_dir PATH/TO/OUTPUT \
    --epochs 50 --patience 10 --n_trials 5
```

Note that `--lr`, `--num_filters`, `--hidden_size`, `--dropout` are optimised
by Optuna in the inner CV loop and not used at their defaults during training.

### Evaluation
We use 5-fold nested CV to produce 20 model checkpoints (5 outer folds × 4 inner
folds). After training, evaluate with:
```
python evaluation/measure_performance.py \
    --out_dir PATH/TO/OUTPUT \
    --data_file data/labeled_sequences.csv
```

The script auto-discovers all `test_outputs_outer*_inner*.pickle` files written by
the training script and reports precision / recall / F1 at tolerances 0–3.

### Predicting
See the [predictor README](predictor/README.md).
