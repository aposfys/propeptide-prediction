# Propeptide prediction
Predicting propeptide cleavage sites in protein sequences from frozen protein language
model embeddings.

Built on [DeepPeptide](https://github.com/fteufel/DeepPeptide) (Teufel et al.,
*Bioinformatics* 2023, [btad616](https://doi.org/10.1093/bioinformatics/btad616)). The
point is to compare protein language models on one task: same head, same data, same
splits, same metric, same training budget, and only the embeddings change. This branch is
the ESM-2 arm (`esm2_t33_650M_UR50D`, 1280 dims).

### Before you run this
This code accompanies an MSc thesis. **If you intend to run it, please contact me first**
— apostolosfysekidis1@gmail.com. The trained weights are not published here and are
available on request. MIT licensed, so this is a request, not a condition.

### Training the model
1. Precompute embeddings using `src/utils/make_embeddings.py`
2. Train the model
```
python run.py --embeddings_dir PATH/TO/EMBEDDINGS -df data/labeled_sequences.csv -pf data/graphpart_assignments.csv --embedding_dim 1280 --epochs 50 --lr 0.0055 --batch_size 20 --dropout 0.6902 --conv_dropout 0.2672 --kernel_size 5 --num_filters 48 --hidden_size 48 --out_dir results/esm2_prop
```
Full procedure, data formats and every hyperparameter in [TRAINING.md](TRAINING.md).

### Results
Propeptide F1 **0.626** at ±3 residue tolerance, above the published propeptide numbers.
See [RESULTS.md](RESULTS.md).

### Branches
One branch per embedding model. See [BRANCHES.md](BRANCHES.md).

### Predicting
[See the predictor README](predictor/README.md)

### License and credit
BSD 3-Clause, inherited from upstream, `Copyright (c) 2023, F Teufel`. The method, dataset
and architecture are Teufel et al.'s; this is a derivative repo for a thesis, not the
reference implementation. Cite the original paper.
