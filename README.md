# DeepPeptide

> **Archived branch, kept for the record.** An older contributed ESM-1b fork with
> different code and a different metric, so its F1 is not comparable to the other
> branches without care. It is preserved because it holds a real baseline
> measurement (propeptide F1 0.371 at ±3). It is not maintained. For the current
> work see `main`, and please contact apostolosfysekidis1@gmail.com before running
> any of it.

Predicting cleaved peptides in protein sequences.

[![DOI](https://zenodo.org/badge/593202385.svg)](https://zenodo.org/badge/latestdoi/593202385)


### Training the model
1. Precompute embeddings using `src/utils/make_embeddings.py`  
2. Train the model  
```
python3 run.py --embeddings_dir PATH/TO/EMBEDDINGS -df data/labeled_sequences.csv -pf data/graphpart_assignments.csv
```
Note that parameters `--lr`, `--batch_size`, `--dropout`, `--conv_dropout`, `--kernel_size`, `--num_filters`, `--hidden_size` were optimized in a nested CV hyperparameter search and not used at their defaults.

### Evaluation
- PeptideLocator was evaluated as a licensed executable and cannot be provided in this repo.
- We used 5-fold nested CV to select 20 model checkpoints trained using `src/train_loop_crf.py`. The selected checkpoints are hardcoded in `evaluation/measure_performance.py`, which computes the performance metrics from the checkpoints' saved predictions.

### Predicting

[See the predictor README](predictor/README.md)