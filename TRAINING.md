# Training

## The model

The original model predicts peptides and propeptides together with 3 labels and 101 CRF
states. Here it's cut down to propeptides only: 2 labels, 51 states (1–50 propeptide, 0
background), and mature-peptide coordinates are dropped. Everything else follows the paper.
Embeddings go into a CNN–biLSTM–CNN, then a linear-chain CRF with a duration-encoded state
grammar (length 5 to 50), decoded with Viterbi. Adam at a constant learning rate, no
scheduler, keep the best checkpoint on validation.

How to reproduce the numbers in [`README.md`](README.md). The commands below are
the ESM-2 arm on `main`; other branches differ only in `--embedding_dim` and the
embedding script.

## 1. Install

```bash
pip install -r requirements.txt   # torch >= 2.0, fair-esm, pandas, numpy, tensorboard, tqdm
```

## 2. Data

Two CSVs, both under `data/` for the UniProt-2022 benchmark:

- `labeled_sequences.csv`, indexed by `protein_id`: `sequence`, `propeptide_coordinates`
  (like `(12-45),(98-113)`), `organism`.
- `graphpart_assignments.csv`, indexed by `AC`: `cluster`, the partition index 0–4 from
  [Graph-Part](https://github.com/graph-part/graph-part).

## 3. Embeddings

```bash
python -m src.utils.make_embeddings data/protein_sequences.fasta PATH/TO/EMBEDDINGS/
```

One `.pt` per sequence, named by the MD5 of the sequence. Existing files are skipped, so
it's safe to interrupt and restart.

## 4. Train

This reproduces the result above:

```bash
python run.py \
    --embeddings_dir PATH/TO/EMBEDDINGS \
    -df data/labeled_sequences.csv -pf data/graphpart_assignments.csv \
    --embedding_dim 1280 --epochs 50 \
    --lr 0.0055 --batch_size 20 --dropout 0.6902 --conv_dropout 0.2672 \
    --kernel_size 5 --num_filters 48 --hidden_size 48 \
    --out_dir results/esm2_prop
```

| argument | used here | notes |
|---|---|---|
| `--embedding_dim` | 1280 | 1536 for ESM3, 1024 for ProstT5 |
| `--epochs` | 50 | |
| `--lr` | 0.0055 | constant, no scheduler. Default is `1e-4` |
| `--batch_size` | 20 | |
| `--dropout` / `--conv_dropout` | 0.6902 / 0.2672 | input and conv dropout |
| `--num_filters` / `--hidden_size` / `--kernel_size` | 48 / 48 / 5 | CNN filters, biLSTM hidden, CNN kernel |
| `--patience` | 0 | early stopping. 0 turns it off, which is upstream behaviour and the default |
| `--out_dir` | | checkpoints, metrics, TensorBoard logs |
