# Next steps — ESM3, multimodal

The question this branch exists to answer: **does conditioning ESM3 on structure
change propeptide cleavage-site prediction?** The existing answer is one run
against one run, and that is not enough to say anything.

## Where this starts

`esm3-propeptide` already carries the multimodal extractor. What it does not
carry is a powered experiment. From RESULTS.md:

| run | n | F1 | precision | recall |
|---|---|---|---|---|
| ESM3 sequence-only, lr 5e-4 bs 20 | 1 | 0.5280 | 0.7179 | 0.4176 |
| ESM3 + structure, lr 5e-4 bs 20 | 1 | 0.4712 | 0.7269 | 0.3486 |

−0.0568, entirely in recall. The widest well-estimated replicate band in the
study is ±0.0409, so this sits just outside what one run each can resolve.
RESULTS.md is right to file it under "suggestive, not established", and right to
list "that structure conditioning hurts" under Not claimed.

Two further structure runs at other learning rates land lower still (0.4654 at
T4, 0.4511 at lr 1e-3). Consistent, also n=1.

So the finding is directionally interesting and evidentially empty. This branch
is for filling it in.

## Why this is a fair question

**Mechanistically:** proprotein convertases cleave at accessible sites. The furin
motif is short and degenerate, and what separates a true site from its many
sequence matches is whether it sits in an exposed, flexible loop — which is what
SASA and backbone geometry encode and a residue sequence does not.

**Methodologically:** a within-model modality swap holds the model, head, data,
splits, metric and budget fixed and changes exactly one thing. Every cross-model
comparison in this study needs a bias direction attached to it (see
RESULTS.md, "What each of these licenses"); this one does not.

**And there is a specific reason to expect a negative result too**, which is why
it is worth doing properly: ESM3's structure tokens are normally *outputs*, and
the authors' own explanation for ESM3's weaker representations is that its late
layers specialise toward its generative objective. Conditioning on structure may
push the representation further toward generation and away from the token-level
discrimination this task needs. A well-powered null, or a well-powered negative,
is a publishable result here. A single run is not.

## What the branch adds

New flags on `src/utils/make_embeddings_esm3_struct.py`:

| flag | what it isolates |
|---|---|
| `--no_coords` | the VQ-VAE structure-token pathway alone |
| `--no_struct_tokens` | Geometric Attention alone |
| `--scramble_structure` | **the negative control** — every structural track present and well-formed, residue axis permuted, no true fold |
| `--scramble_seed` | reproducibility for the above, mixed per sequence hash |

The two pathways matter because ESM3 has two, and the existing structure run fed
both. Hayes et al. are explicit that coordinates go through Geometric Attention
and are "not embedded", while structure tokens "are generally used as model
outputs" — so they are not two views of the same thing, and an effect could come
from either.

Also on this branch, and not specific to structure:

- **Both boundary tolerances.** Every run writes `f1 propeptides@1` and
  `f1 propeptides@3`. The unsuffixed `f1 propeptides` still holds ±3 and model
  selection still uses ±3, so nothing in RESULTS.md moves.
- **`valid_metrics.json` on the single-run path.** RESULTS.md, Provenance: 68 of
  84 finished runs have no record of which epoch was selected, because `train()`
  wrote test metrics and not validation ones. Fixed, mirroring
  `main:src/train_loop_crf.py:125`. Every run launched from here is auditable.
- **A configurable CRF grammar** — see [GRAMMAR.md](GRAMMAR.md).

## The runs, in order

All at ESM3's own tuned configuration (`lr 5e-4, bs 70`), or at `lr 5e-4, bs 20`
if you want to extend the existing structure pair directly. Pick one and keep it
for every arm.

### 1. Extract (GPU)

```bash
python -m src.utils.fetch_afdb_structures --out_dir structures/

# sequence-only, through this exact code path -- isolates the track
# contribution from any other difference between the two scripts
python -m src.utils.make_embeddings_esm3_struct \
    --structures_dir structures/ --out_dir embeddings/esm3_seqonly --no_structure

# all tracks (the existing treatment)
python -m src.utils.make_embeddings_esm3_struct \
    --structures_dir structures/ --out_dir embeddings/esm3_struct \
    --gpu_max_len 2000 --max_struct_len 1024

# the two pathways, separately
python -m src.utils.make_embeddings_esm3_struct \
    --structures_dir structures/ --out_dir embeddings/esm3_tokens_only \
    --no_coords --gpu_max_len 2000 --max_struct_len 1024
python -m src.utils.make_embeddings_esm3_struct \
    --structures_dir structures/ --out_dir embeddings/esm3_geom_only \
    --no_struct_tokens --gpu_max_len 2000 --max_struct_len 1024

# the control
python -m src.utils.make_embeddings_esm3_struct \
    --structures_dir structures/ --out_dir embeddings/esm3_struct_scrambled \
    --scramble_structure --gpu_max_len 2000 --max_struct_len 1024
```

Run `preflight.py` on each before training. It refuses mis-scaled embeddings,
which is the failure that invalidated every pre-2026-08-19 ESM3 result — see
[EMBEDDINGS.md](EMBEDDINGS.md).

### 2. Replicates

Powered at 80%, two-sided α = 0.05, on the ESM3 tuned group's own replicate sd of
0.0185 (n=15, the best-estimated group in the study):

| effect to detect | replicates per arm |
|---|---|
| 0.057 (the observed structure effect) | 2 |
| 0.05 | 4 |
| 0.03 | 7 |
| 0.02 | 15 |

**Run 8 per arm.** That resolves 0.03 with margin. It is deliberately more than
the 4 the observed effect needs, because the observed effect is the thing under
test and assuming its size to size the test is circular. Structure runs may also
be noisier than the sequence-only group, and 8 leaves room for that.

Five arms × 8 = 40 runs.

### 3. Read it in this order

1. **all tracks vs scrambled.** The result. Isolates geometry from "having a
   second track at all".
2. **all tracks vs sequence-only-through-this-path.** The effect as RESULTS.md
   currently frames it, now with n=8.
3. **tokens-only vs geometry-only.** Only worth interpreting if (1) shows
   something. If it does, this says which pathway carries it.

Report ±1 alongside ±3. The existing structure deficit is *entirely in recall*
(precision 0.7269 vs 0.7179, essentially unchanged; recall 0.3486 vs 0.4176). If
structure is doing anything real to boundary placement rather than to detection,
±1 is where it shows up, and the current ±3-only reporting cannot see it.

## What would make the result unpublishable

- **AlphaFold models the precursor.** AFDB predicts the full UniProt sequence
  including the propeptide, and propeptide regions are frequently low-confidence
  and disordered. A structure effect may be a disorder-prediction effect wearing
  a different name. Correlate the per-run gain with per-residue pLDDT in the
  propeptide span before claiming otherwise.
- **~12% of proteins have no structure and are masked.** They are kept so the
  partitions stay identical, but a mask is itself a signal the head can learn.
  The scrambled control shares the mask pattern, which is what bounds this.
- **The learning rate was tuned for sequence-only ESM3.** A structure-conditioned
  representation may want a different one, so a deficit at a fixed lr is partly a
  transfer failure. Cheapest guard: run the scrambled control at the same lr, and
  compare treatment against control rather than against the tuned baseline.

## On finding a bigger ESM3

Short answer: **there isn't one, and this is not the axis to spend on.**

`esm3_sm_open_v1` (1.4B) is the only ESM3 with open weights. The 7B and 98B
models are API-only through EvolutionaryScale's Forge — not downloadable, not
free, and not reproducible from a repository, which makes them a poor fit for a
thesis arm that has to be re-runnable.

The open alternatives, and what each would actually test:

| model | open | input | what a new arm would tell you |
|---|---|---|---|
| ESM C 300M / 600M | yes, non-commercial | sequence | scale within a newer sequence-only family |
| ESM-2 3B / 15B | yes | sequence | whether ESM-2's 0.6153 is a scale effect |
| **SaProt 650M** | yes | **sequence + 3Di jointly tokenised** | the multimodal question, done properly |
| ProstT5 | yes | sequence or 3Di | already an arm; structure version on `prost5-multimodal` |

**SaProt is the one worth adding.** Its vocabulary is the 35 × 20 product of
residue and structure token, so structure enters at the token level rather than
by conditioning or concatenation — a genuinely different way of asking the same
question, at a size comparable to what is already here. The 3Di from
`prost5-multimodal`'s `make_3di.py` feeds it directly.

**ESM-2 15B is the one to skip.** RESULTS.md already shows that once tuning
budget is equalised the models stop separating, so a bigger sequence-only model
mostly re-tests the finding that scale is not the bottleneck — at roughly 23×
the embedding storage and a much heavier extraction.

The honest framing for the thesis: this study is about *what information the
representation carries*, not about scale. Adding a bigger sequence-only model
answers a question the study is not asking.
