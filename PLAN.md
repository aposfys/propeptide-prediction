# The plan

The dataset is the published one. `data/labeled_sequences.csv` and
`data/graphpart_assignments.csv`, unchanged. Where this disagrees with anything
else in this repository, this wins.

## The grammar is 101 states

**Every arm uses 101 states** (`--max_peptide_len 100`): one background state
plus propeptide positions 1–100, two labels.

That is DeepPeptide's own state budget — their model is 1 background + 50
peptide + 50 propeptide — reallocated so every non-background state models
propeptides. It matters because the published model has 101 states and the
propeptide-only fork cut it to 51. Comparing a 51-state model against a
101-state published one gives the new model a smaller state space and makes the
comparison unequal in the wrong direction. Matching the budget removes that.

**What it costs, stated plainly.** All the benchmark's propeptides are 5–50, so
states 51–100 receive no positive training signal — the partition function
suppresses them and nothing teaches them. Decoding is 3.9× slower. Whether that
costs accuracy is exactly what the ablation in step 1 measures, on identical
data, and it is the only thing that could send this decision back.

**And it means every arm must be re-run at 101 states.** The existing ProstT5
group (n=5) and the ESM3 runs are at 51 states and do not carry over. That is
the price of the grammar choice and it is 46 runs rather than 36.

## The design

Three arms. Same head, same 51-state grammar, same T4 hyperparameters, same
data, same splits. Only the embeddings change.

| arm | input | dims | status |
|---|---|---|---|
| **1 baseline** | ESM-2, sequence only | 1280 | **running as `esm2_g101_rep*`** |
| 2 | ESM3, sequence + structure tracks | 1536 | to run |
| 2c | ESM3, sequence only | 1536 | control for arm 2 |
| 3 | ProstT5, sequence + 3Di | 1024 | to run |
| 3c | ProstT5, sequence only | 1024 | control for arm 3 |

**The number to beat is arm 1 at 101 states**, which is the job in flight. Not
the 51-state 0.6153 — comparing a 101-state arm against a 51-state figure would
confound the grammar with the representation.

## What the 51-state runs are now for

They are **reference, not baseline**. Nothing at 51 states can be compared
directly against a 101-state arm.

| 51-state group | n | mean | sd | use |
|---|---|---|---|---|
| ESM-2 sequence-only | 10 | 0.6153 | 0.0242 | the grammar ablation's control |
| ProstT5 sequence-only | 5 | 0.5189 | 0.0306 | reference only |
| ESM3 sequence-only @ T4 | 1 | 0.5270 | — | reference only |
| ESM3 + structure @ T4 | 1 | 0.4654 | — | reference only |

Their value is that the ESM-2 pair, 51 versus 101 states on identical data,
isolates the grammar's cost — which no other comparison in the study can do.

## Steps

### Step 0 — per-mechanism rescoring · DONE

Convertase 0.7073, zymogen 0.6813, unassigned 0.5503 at ±3, against a
within-class sd of 0.021–0.040. At ±1 the order reverses and convertase becomes
the worst class at 0.3791. The pooled 0.6153 describes no subgroup.

**The paper keeps this whatever else happens.** It needs no new runs and it is
the spine if every structure arm fails.

### Step 1 — the baseline, and the grammar ablation, are the same runs · 10 runs · IN FLIGHT

```bash
bash run_grammar_ablation.sh 10 100 results/esm2_rep1/config.json
```

Extend the four in flight to ten. These runs do double duty: against the
51-state `esm2_rep*` they measure what the grammar costs, and on their own they
are **arm 1**, the number every other arm has to beat.

**Gate.** If the 101-state group falls well below 0.6153 — more than the
±0.0574 single-run band — the grammar is buying nothing and costing accuracy on
data that cannot use it. That is the one result that sends the 101-state
decision back for review. Otherwise it stands and everything below runs at 101.

### Step 2 — structures and 3Di · CPU only

```bash
pip install mini3di biotite
python -m src.utils.fetch_afdb_structures --out_dir structures/
python -m src.utils.make_3di --structures_dir structures/ --out_dir three_di/
```

**Gate.** `make_3di` must report ≥80% `ok`.

### Step 3 — the two sequence-only controls · 16 runs

ESM3 and ProstT5, sequence only, **at 101 states**, T4. Their existing groups are
at 51 states and do not transfer. Without these, neither structure arm can be
read as anything but a loss to the baseline.

Embeddings for both already exist on disk, so this is GPU time only.

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

8 replicates each, at T4, **101 states**.

**Gate.**

- **either arm ≥ ~0.650** → the headline. Go to step 5.
- **neither beats 0.6153, but one beats its own sequence-only control** →
  structure helps, this representation does not win. Publishable, and honest.
  Go to step 5.
- **neither beats 0.6153 nor its own control** → structure does not move this
  task. **Stop.** The paper is step 0 plus a controlled negative. Skip to step 6.

Read each structure arm against **its own 101-state control from step 3**, not
only against arm 1. At 51 states ESM3 and ProstT5 both sit ~0.09 behind ESM-2, so
a structure gain and a loss to the baseline are entirely compatible, and only the
within-arm comparison separates them.

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
| 1 baseline / grammar ablation | 10, 4 in flight | yes |
| 2 structures + 3Di | 0 | no |
| 3 ESM3 + ProstT5 sequence-only controls | 16 | yes |
| 4 structure arms | 16 | yes |
| 5 scrambled controls | 8 | only for the arm that moved |

46 runs. Keeping the published dataset saved nothing here, because moving to 101
states means every arm is re-run; what it saved is the argument.

## Not doing, and why

- **Rebuilding the dataset.** Dropped. The gain was +22% positives before
  GraphPart, and only 287 proteins of that are genuinely new UniProt entries;
  most of the rest exists only if the window widens. Breaking comparability with
  every published number for that is not a trade worth making. The analysis stays
  on file in `DATASET_V2_VALIDATION.md` and is worth citing as a **limitation**:
  the published benchmark covers 57.1% of eligible propeptide proteins and
  carries 680 real propeptides its labels omit.
- **Widening the label window.** The 5..50 data stays; only the grammar widens.
- **ESM-2 + 3Di.** The likeliest arm to clear 0.650, needing +0.035 where ESM3
  and ProstT5 need ~+0.13. Excluded because the design is three clean
  representation arms. `make_hybrid_embeddings.py` is here if that is revisited.
- SaProt, equal-budget Optuna. Later.
