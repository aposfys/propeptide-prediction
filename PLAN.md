# The plan

The dataset is the published one. `data/labeled_sequences.csv` and
`data/graphpart_assignments.csv`, unchanged. Where this disagrees with anything
else in this repository, this wins.

## Consequence you have to accept: back to 51 states

Every propeptide in the published benchmark is 5–50 residues. A 101-state
grammar leaves states 51–100 unvisited by any label, costs 3.9× decode, and
buys nothing. **The grammar goes back to 51 states**, which is also the faithful
setting for a propeptide-only fork of a 101-state joint model.

The 101-state question is not dead, it is deferred. It becomes live again only
if the dataset ever widens, and `PAPER.md` and `DATASET_V2_VALIDATION.md` keep
that analysis on file.

## The design

Three arms. Same head, same 51-state grammar, same T4 hyperparameters, same
data, same splits. Only the embeddings change.

| arm | input | dims | status |
|---|---|---|---|
| **1 baseline** | ESM-2, sequence only | 1280 | **done: 0.6153, sd 0.0242, n=10** |
| 2 | ESM3, sequence + structure tracks | 1536 | to run |
| 3 | ProstT5, sequence + 3Di | 1024 | to run |

**The number to beat is 0.6153.** At 8 replicates a resolvable win needs ~0.650.

## What already exists, and what that saves

| arm | n | mean | sd |
|---|---|---|---|
| ESM-2 sequence-only | 10 | 0.6153 | 0.0242 |
| ProstT5 sequence-only | 5 | 0.5189 | 0.0306 |
| ESM3 sequence-only @ T4 | **1** | 0.5270 | — |
| ESM3 + structure @ T4 | **1** | 0.4654 | — |

The baseline and the ProstT5 control are already done. **ESM3 has no replicated
T4 arm**, so its control has to be run or its structure result cannot be read.

## Steps

### Step 0 — per-mechanism rescoring · DONE

Convertase 0.7073, zymogen 0.6813, unassigned 0.5503 at ±3, against a
within-class sd of 0.021–0.040. At ±1 the order reverses and convertase becomes
the worst class at 0.3791. The pooled 0.6153 describes no subgroup.

**The paper keeps this whatever else happens.** It needs no new runs and it is
the spine if every structure arm fails.

### Step 1 — let the grammar ablation finish · 4 runs · IN FLIGHT

Do **not** extend it to 10. It no longer feeds the plan. Four replicates are
enough to record whether a duration-coded CRF costs accuracy when its extra
states cannot be used, which is a methodological note worth one paragraph.

### Step 2 — structures and 3Di · CPU only

```bash
pip install mini3di biotite
python -m src.utils.fetch_afdb_structures --out_dir structures/
python -m src.utils.make_3di --structures_dir structures/ --out_dir three_di/
```

**Gate.** `make_3di` must report ≥80% `ok`.

### Step 3 — the ESM3 control · 8 runs

ESM3 sequence-only at T4, 51 states. It exists at n=1; without replicates the
arm-2 result is unreadable.

### Step 4 — the two structure arms · 16 runs

```bash
python -m src.utils.make_embeddings_esm3_struct \
    --structures_dir structures/ --out_dir embeddings/esm3_struct \
    --gpu_max_len 2000 --max_struct_len 1024

python -m src.utils.make_embeddings_prost5_struct \
    data/protein_sequences.fasta embeddings/prost5_fused \
    --tracks aa+3di --fuse renorm --three_di three_di/three_di.json
```

`--fuse renorm` holds ProstT5 at 1024 dims so its head matches its own control
exactly. ESM3 is 1536 whatever it is fed.

8 replicates each, at T4, 51 states.

**Gate.**

- **either arm ≥ ~0.650** → the headline. Go to step 5.
- **neither beats 0.6153, but one beats its own sequence-only control** →
  structure helps, this representation does not win. Publishable, and honest.
  Go to step 5.
- **neither beats 0.6153 nor its own control** → structure does not move this
  task. **Stop.** The paper is step 0 plus a controlled negative. Skip to step 6.

Read arm 2 against 0.5270 and arm 3 against 0.5189, not only against the
baseline. Both start ~0.09 behind ESM-2, so a structure gain and a loss to the
baseline are entirely compatible and the within-arm comparison is what separates
them.

### Step 5 — scrambled controls and per-mechanism breakdown · 8 runs + no GPU

Only for whichever arm moved.

```bash
python -m src.utils.make_embeddings_esm3_struct ... --scramble_structure
python -m src.utils.make_embeddings_prost5_struct ... --shuffle_3di
python -m src.utils.score_by_mechanism results/<arm>_rep*/test_outputs.pickle
```

Beating the sequence-only control says the extra track helped. Beating the
**scrambled** control says *structure* helped. Only the second is a mechanistic
claim.

### Step 6 — write up whichever is true

## Run budget

| step | runs | GPU |
|---|---|---|
| 1 grammar ablation | 4, in flight | yes |
| 2 structures + 3Di | 0 | no |
| 3 ESM3 control | 8 | yes |
| 4 structure arms | 16 | yes |
| 5 scrambled controls | 8 | only for the arm that moved |

36 runs, down from 54, because the published dataset means the baseline and the
ProstT5 control are already in hand.

## Not doing, and why

- **Rebuilding the dataset.** Dropped. The gain was +22% positives before
  GraphPart, and only 287 proteins of that are genuinely new UniProt entries;
  most of the rest exists only if the window widens. Breaking comparability with
  every published number for that is not a trade worth making. The analysis stays
  on file in `DATASET_V2_VALIDATION.md` and is worth citing as a **limitation**:
  the published benchmark covers 57.1% of eligible propeptide proteins and
  carries 680 real propeptides its labels omit.
- **101 states.** Deferred with the dataset.
- **ESM-2 + 3Di.** The likeliest arm to clear 0.650, needing +0.035 where ESM3
  and ProstT5 need ~+0.13. Excluded because the design is three clean
  representation arms. `make_hybrid_embeddings.py` is here if that is revisited.
- SaProt, equal-budget Optuna. Later.
