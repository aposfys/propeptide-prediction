# Next steps — ProstT5, structure-conditioned

The question this branch exists to answer: **does giving ProstT5 the structural
language it was trained on change propeptide cleavage-site prediction?**

The sequence-only ProstT5 arm reached F1 0.5189 (sd 0.0306, n=5) at ESM-2's
hyperparameters, statistically indistinguishable from ESM3 (p = 0.926) and below
ESM-2 (0.6153). It used `<AA2fold>` and amino acids only. ProstT5's other half —
Foldseek 3Di, the language that makes it "bilingual" — was never fed to it.

## Why this is a fair question and not a fishing trip

Two reasons, and it matters that they are different.

**Mechanistically, structure is the right kind of extra evidence.** Proprotein
convertases cleave at accessible sites: the classic furin motif is short and
degenerate, and what separates a real site from the many sequence matches is
whether the site sits in an exposed, flexible loop. That is exactly what a
structural alphabet encodes and a residue sequence does not. The hypothesis has a
mechanism behind it.

**Methodologically, a within-model modality swap is the strongest design here.**
RESULTS.md already shows why: cross-model comparisons in this study are confounded
by tuning budget, and the honest reading of ESM3-vs-ProstT5 (p = 0.926) is that
once each model is near its own best configuration, the models stop separating.
Adding a modality *inside one model* holds the model, the head, the data, the
splits, the metric and the budget fixed and changes one thing. It is the only
contrast in this repository that does not need a bias direction attached to it.

## What the branch adds

| file | what it does |
|---|---|
| `src/utils/fetch_afdb_structures.py` | AlphaFold DB models for every accession, with a manifest. Ported unchanged from `esm3-propeptide` so both structure arms rest on the same structures. |
| `src/utils/make_3di.py` | 3Di strings from those structures, via `mini3di` (pure Python) or the Foldseek binary. |
| `src/utils/make_embeddings_prost5_struct.py` | ProstT5 over amino acids and 3Di, concatenated to 2048 dims. `--tracks` selects the arm, `--shuffle_3di` is the control. |
| `src/utils/propeptide_length_audit.py` | The CRF grammar sizing analysis. See [GRAMMAR.md](GRAMMAR.md). |
| `test_grammar.py` | Asserts the configurable grammar reproduces the published one exactly at its defaults. |

## The runs, in order

Everything below is a single `run.py` invocation per replicate at ProstT5's
existing configuration (T4: `lr 0.0055, bs 20, dropout 0.6902, conv_dropout
0.2672, kernel_size 5, num_filters 48, hidden_size 48, 50 epochs`), because the
whole point is to change the embedding and nothing else.

### 0. Build the inputs (once, ~2–4 h, mostly download)

```bash
pip install mini3di biotite
python -m src.utils.fetch_afdb_structures --out_dir structures/
python -m src.utils.make_3di --structures_dir structures/ --out_dir three_di/
```

Check the `make_3di` summary before going further. Expect roughly 85–90% `ok`.
If it is near zero, the manifest is wrong and everything downstream is a
sequence-only run wearing a structure arm's name.

### 1. Extract the three embedding sets (GPU, ~1–2 h each)

```bash
# the AA arm, which must reproduce the existing sequence-only embeddings
python -m src.utils.make_embeddings_prost5_struct \
    data/protein_sequences.fasta embeddings/prost5_aa --tracks aa

# structure only
python -m src.utils.make_embeddings_prost5_struct \
    data/protein_sequences.fasta embeddings/prost5_3di \
    --tracks 3di --three_di three_di/three_di.json

# both, fused at 1024 dims so the model stays byte-identical to the
# sequence-only arm -- see FUSION.md for why renorm rather than sum or mean
python -m src.utils.make_embeddings_prost5_struct \
    data/protein_sequences.fasta embeddings/prost5_fused \
    --tracks aa+3di --fuse renorm --three_di three_di/three_di.json

# the control: same architecture, same mask pattern, wrong structures
python -m src.utils.make_embeddings_prost5_struct \
    data/protein_sequences.fasta embeddings/prost5_fused_shuf \
    --tracks aa+3di --fuse renorm --three_di three_di/three_di.json --shuffle_3di
```

Then, before training anything:

```bash
python -m src.utils.verify_embeddings --embeddings_dir embeddings/prost5_aa3di \
    --embedding_dim 2048
```

**Sanity check that is worth the five minutes:** the first 1024 dims of
`prost5_aa3di` must equal `prost5_aa` exactly. If they do not, the two arms
differ by more than structure and nothing below is interpretable. The mock-encoder
test asserts this property of the code; this asserts it of the actual tensors.

### 2. Replicates, not single runs

This is the part the existing structure result got wrong. RESULTS.md reports
ESM3 + structure at −0.0568 against sequence-only, **n = 1 each**, against a
±0.0409 prediction band. That is not a measurement.

Powered at 80%, two-sided α = 0.05, using ProstT5's own replicate sd of 0.0306:

| effect to detect | replicates per arm |
|---|---|
| 0.057 (the ESM3 structure effect) | 6 |
| 0.05 | 7 |
| 0.03 | 18 |
| 0.02 | 38 |

**Run 8 per arm.** That covers the 0.05 effect with margin and costs 32 runs
across four arms. Detecting 0.02 is out of reach at this variance and should not
be attempted — say so rather than running 38 and calling it exploratory.

Every arm below is 1024 dims, so `--embedding_dim` is the same for all of them
and the model is the original one throughout.

```bash
for arm in prost5_aa prost5_3di prost5_fused prost5_fused_shuf; do
  dim=1024
  for rep in $(seq 1 8); do
    python run.py --embeddings_dir embeddings/$arm \
      -df data/labeled_sequences.csv -pf data/graphpart_assignments.csv \
      --embedding_dim $dim --epochs 50 --lr 0.0055 --batch_size 20 \
      --dropout 0.6902 --conv_dropout 0.2672 --kernel_size 5 \
      --num_filters 48 --hidden_size 48 \
      --out_dir results/${arm}_rep${rep}
  done
done
```

### 3. Read it in this order

1. **fused vs fused-shuffled.** This is the result. It is the only pair that
   isolates structural *information* rather than the presence of a second
   channel. With `--fuse renorm` both arms are 1024 dims and the model is the
   original one, so there is no capacity difference anywhere in the comparison.
2. **`aa+3di` vs `aa`.** Real but weaker: 2048 dims against 1024 changes the
   head's capacity as well as its information.
3. **`3di` vs `aa`.** How much of the task is structural at all. Expect it to
   lose; if it does not, that is the interesting finding.

Report both tolerances. Every run now writes `f1 propeptides@1` and
`f1 propeptides@3` into `test_metrics.json`, with the bare `f1 propeptides` key
still holding ±3 so existing readers and every number in RESULTS.md stay valid.

±1 is where a structural signal should show up first if it is real. Structure
constrains *where* a cleavage site can be, so it should sharpen boundary
placement before it changes whether a site is found at all. A gain that appears
at ±3 but not at ±1 is more likely extra head capacity than better geometry.

## Keeping the original architecture

`--fuse renorm` keeps the input at 1024 dims, so every arm above builds a model
byte-identical to `prost5-propeptide` — same 51-state CRF, same constraint mask,
192,369 parameters. `test_architecture.py` asserts it.

`--fuse concat` is the 2048-dim version. It widens `conv1` by 98,304 parameters
and changes nothing else. It is worth running as a second experiment, not as the
primary one, because a gain there is confounded with head capacity in a way the
fused arms are not. [FUSION.md](FUSION.md) has the trade-off in full.

## What would make this branch's result unpublishable

State these before running, not after.

- **The zero-mask confound.** ~12% of proteins get an all-zero 3Di channel, a
  value the encoder never emits, so the head can learn to detect "no structure".
  The shuffled control has the same mask pattern, which bounds this — but if
  `aa+3di` beats `aa` while `aa+3di_shuffled` also beats `aa`, the mask is doing
  the work, not the structure. Check that first.
- **AlphaFold models the precursor.** AFDB predicts the full UniProt sequence,
  propeptide included, and propeptide regions are frequently low-confidence and
  disordered. A "structure" effect may really be a disorder-prediction effect.
  That is still a result, but it is a different one, and it should be named.
  Correlating per-residue pLDDT with the gain is the cheap way to check.
- **3Di is not the structure.** It is a 20-letter quantisation of local
  environment. A null result bounds what 3Di carries, not what structure carries.

## After this

- **SaProt** (650M, open weights) is the obvious next arm: its vocabulary is the
  35 × 20 product of residue and 3Di token, so structure enters at the token
  level rather than by concatenating two encoder passes. If concatenation here
  shows anything, SaProt is how to do it properly, and the 3Di from `make_3di.py`
  feeds it directly.
- Not ESM3-large. See the ESM3 branch's `NEXT_STEPS.md` — there is no larger
  open-weights ESM3.
