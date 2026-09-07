# Keeping the original architecture

**The model is unchanged.** At its defaults this branch builds a byte-identical
model to `prost5-propeptide`: 51 CRF states, 99 transitions, the same constraint
mask (SHA `b0a6f981…`), the same 20 state-dict tensors, 192,369 trainable
parameters. `test_architecture.py` asserts every one of those against values
measured on the parent branch.

```bash
python test_architecture.py
```

There is exactly **one** way to change it, and it is opt-in.

## The one change: concatenating two channels

`--fuse concat` stacks the amino-acid and 3Di embeddings into 2048 dims. That
widens the first convolution and nothing else:

| | 1024 dims | 2048 dims |
|---|---|---|
| `conv1.weight` | [32, 1024, 3] | [32, 2048, 3] |
| trainable parameters | 192,369 | 290,673 |
| every other tensor | unchanged | unchanged |

+98,304 parameters, +51% on the head. The biLSTM, the second convolution, the
emission layer and the CRF are untouched.

## How to avoid it

Three of the four fusion modes keep the input at 1024 dims, so the model stays
byte-identical to the sequence-only arm and the only thing that differs between
the arms is the features.

| `--fuse` | dims | conv1 | per-token norm, with structure | without |
|---|---|---|---|---|
| `concat` | 2048 | **widened** | 1.42× | 1.00× |
| **`renorm`** | 1024 | unchanged | **1.00×** | **1.00×** |
| `sum` | 1024 | unchanged | 1.43× | 1.00× |
| `mean` | 1024 | unchanged | 0.72× | 1.00× |

Norms are measured over two roughly uncorrelated channels; the last column is
the ~12% of proteins with no usable AlphaFold model, whose 3Di channel is masked.

**Use `renorm`.** It adds the two channels and rescales each token to the amino
acid channel's own norm, so the direction of the vector carries the structural
information and the magnitude carries nothing. Raw `sum` and `mean` leave the
unstructured 12% at a different scale from the rest, which splits the dataset
into two scale regimes inside a head that has no input normalisation. That is
the same class of defect as the ESM3 pre-LayerNorm bug in
[EMBEDDINGS.md](EMBEDDINGS.md), which saturated 90.7% of the biLSTM gates and
invalidated every ESM3 result before 2026-08-19. Do not repeat it on purpose.

Element-wise fusion is defensible here in a way it usually is not: both channels
are outputs of the *same* ProstT5 encoder, so they already live in one vector
space. Adding embeddings from two different models would not be.

## The cleanest option of all

`--tracks 3di` is 1024 dims of pure structure. Same architecture, same scale, no
fusion, no dimensionality question, no capacity confound. It answers "does
structure carry propeptide signal at all?" with nothing to argue about.

It cannot answer "does structure ADD to sequence?", which is why the fused arms
exist. But if you want one structure result with no methodological asterisk, it
is that one.

## What to run

```bash
# structure only -- architecture identical, cleanest single contrast
python -m src.utils.make_embeddings_prost5_struct \
    data/protein_sequences.fasta embeddings/prost5_3di \
    --tracks 3di --three_di three_di/three_di.json

# sequence + structure, architecture identical
python -m src.utils.make_embeddings_prost5_struct \
    data/protein_sequences.fasta embeddings/prost5_fused \
    --tracks aa+3di --fuse renorm --three_di three_di/three_di.json

# the control for it, same architecture, wrong structures
python -m src.utils.make_embeddings_prost5_struct \
    data/protein_sequences.fasta embeddings/prost5_fused_shuf \
    --tracks aa+3di --fuse renorm --three_di three_di/three_di.json --shuffle_3di
```

All three take `--embedding_dim 1024`, exactly as the sequence-only arm does.

## A note on what `renorm` costs

Fusing into 1024 dims is architecturally free but not informationally free. Two
1024-dim vectors added together cannot be separated again, so the head can no
longer weight sequence and structure independently the way it can with
concatenation. If structure helps under `concat` and not under `renorm`, that is
informative rather than contradictory: it says the head needed the extra
capacity to use it, which is a weaker claim than "structure carries the signal".

Running both is 16 extra runs and settles it. Running only `renorm` keeps the
architecture argument clean and risks a false negative. That trade is a choice,
not an oversight, and whichever you pick should be stated.

## ESM3 has none of this problem

`esm3-multimodal` outputs 1536 dims whatever tracks it is fed, because the
structural information enters through ESM3's own conditioning rather than by
concatenation. Its model is byte-identical to `esm3-propeptide` — 241,521
parameters, same CRF mask — for every arm including the scrambled control. There
is no fusion decision to make there.
