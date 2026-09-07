# The pipeline, in order

Linear. Each step needs only the one before it.

```
1  build the dataset        CPU     done, shipped in data_v4/
2  extract ESM-2 embeddings GPU     all 9,759 proteins
3  motif clusters           CPU     needs step 2
4  GraphPart                CPU     needs step 3, ~3 h
5  the other two embedders  GPU     ESM3 + structure, ProstT5 + 3Di
6  train the three arms     GPU     needs steps 4 and 5
```

There is no circular dependency. I implied one earlier by running GraphPart
before the embeddings existed, which forced a fallback label. That was my
sequencing mistake, not a property of the pipeline.

## 1. The dataset — already built

`data_v4/` is in this branch: 9,759 proteins, 8,516 with a propeptide, 9,615
spans, all inside 5..100. Built under Teufel et al.'s published protocol with the
length window widened. Rebuild it yourself with:

```bash
python -m src.utils.build_dataset --out_dir data_v4 --max_len 100
python -m src.utils.validate_dataset --data_dir data_v4 --max_len 100
```

The validator must print `0 failure(s)`. See [COVERAGE.md](COVERAGE.md).

## 2. ESM-2 embeddings for the whole set

```bash
python -m src.utils.make_embeddings data_v4/protein_sequences.fasta \
    embeddings/v4_esm2
```

All 9,759, not just the 1,674 that are new. Keyed by md5 of the sequence, so a
protein shared with the old benchmark produces the identical file and nothing is
wasted if you point at an existing directory first.

## 3. Motif clusters — Teufel's balancing label

```bash
python -m src.utils.make_motif_clusters \
    --data_file data_v4/labeled_sequences.csv \
    --embeddings_dir embeddings/v4_esm2 \
    --fasta_in data_v4/protein_sequences.fasta \
    --fasta_out data_v4/protein_sequences_clustered.fasta
```

Two residues before the N terminus and after the C terminus of each propeptide,
each encoded with ESM-2, the four vectors concatenated, k-means at k=50. That is
what `motif_cluster=13` means in the distributed FASTA headers. CPU only, minutes.

## 4. GraphPart

```bash
graphpart needle -ff data_v4/protein_sequences_clustered.fasta \
    -th 0.3 -pa 5 -ln motif_cluster -nm -nt 10 \
    -of data_v4/graphpart_assignments.csv
```

Published settings, unchanged: Needleman–Wunsch, 30% identity, 5 partitions,
`--no-moving`. 95.2 million pairwise alignments; about 3 hours on 10 cores,
measured from two timing points.

Installing the toolchain, since neither ships by default:

```bash
pip install graph-part
micromamba create -y -p ./embossenv -c bioconda -c conda-forge emboss
export PATH="$PWD/embossenv/bin:$PATH"
```

**Gate.** Retention ≥85% — the published file kept 90.2%, 7,623 of 8,449 — and
`between_connectivity` 0 on every row, which is what proves 30% separation
actually held. Also check the 275 duplicate sequences did not straddle a
partition.

## 5. The two structure embedders

```bash
python -m src.utils.fetch_afdb_structures --data_file data_v4/labeled_sequences.csv \
    --out_dir structures/

python -m src.utils.make_embeddings_esm3_struct --data_file data_v4/labeled_sequences.csv \
    --structures_dir structures/ --out_dir embeddings/v4_esm3_struct \
    --gpu_max_len 2000 --max_struct_len 1024

python -m src.utils.make_3di --data_file data_v4/labeled_sequences.csv \
    --structures_dir structures/ --out_dir three_di/
python -m src.utils.make_embeddings_prost5_struct \
    data_v4/protein_sequences.fasta embeddings/v4_prost5_fused \
    --tracks aa+3di --fuse renorm --three_di three_di/three_di.json
```

Each arm also needs its own sequence-only control, or a result reads only as a
loss to the baseline rather than as a structure effect.

## 6. Train

From each arm's branch, 8 replicates:

```bash
ARM=esm2 EMB=embeddings/v4_esm2 DIM=1280 MAX_LEN=100 bash run_arm.sh 8
```

`MAX_LEN=100` gives the 101-state grammar, which this dataset actually populates
— every state from 4 to 100 is now reached by real labels, which was not true of
the published benchmark.

Point `--data_file` and `--partitioning_file` at `data_v4/`.

## What is running right now, and why it will be superseded

A GraphPart run on `data_v4` balanced on `mechanism` instead of `motif_cluster`,
because I started it before step 2 existed. It gives the retention number about
three hours early, which de-risks the gate. **Discard its partition** and use the
one from step 4.
