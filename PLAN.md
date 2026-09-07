# The plan

Three representation arms, one grammar, one head, one branch. Steps run in
order, each has a gate, and a failed gate has a written consequence.

Where this disagrees with `PAPER.md`, `GRAMMAR.md`, `DATASET_V2.md` or either
`NEXT_STEPS.md`, this wins.

## The design

| arm | input | dims | embeddings |
|---|---|---|---|
| **1 baseline** | ESM-2, sequence only | 1280 | already on disk |
| 2 | ESM3, sequence + structure tracks | 1536 | `make_embeddings_esm3_struct.py` |
| 3 | ProstT5, sequence + 3Di | 1024 or 2048 | `make_embeddings_prost5_struct.py` |

Every arm: **101 CRF states** (`--max_peptide_len 100`), propeptide-only labels
(background, propeptide), T4 hyperparameters, same data, same splits, same head.
Only the embeddings change.

## The number to beat

**ESM-2, sequence only, 101 states, T4.** That run is in flight now as
`esm2_g101_rep1..4`. It is the baseline because it is the only one that differs
from the other arms in exactly one thing — the representation.

The 51-state figure of 0.6153 is **not** the target. Comparing a 101-state arm
against it would confound grammar with representation.

Fill the baseline to 8 replicates before reading any other arm against it.

## One branch, not three

This is `paper-3arm`, and all three arms run from it.

`esm3-multimodal` and `prost5-multimodal` have **different training code** —
`train_loop_crf.py`, `crf_models.py`, `dataset.py` and `manuscript_metrics.py`
all differ between them. Running arm 3 on one branch and arms 1–2 on the other
would vary the training loop alongside the representation, and no amount of
replication fixes that. `RESULTS.md` already warns against comparing F1 across
branches.

So this branch carries every extractor, and the arms are told apart by
`--embeddings_dir`, never by which code produced them.

## Each arm needs its own sequence-only control

Arms 2 and 3 get **two** comparisons, and the second is what makes the work
publishable if the first fails:

- **against arm 1** — is this the best representation? Your bar.
- **against its own sequence-only self at 101 states** — did structure help?

At 51 states, ESM3 sequence-only scores 0.5227 and ProstT5 0.5189, against
ESM-2's 0.6153. Both start about 0.09 behind. If ESM3 + structure lands at 0.55
it loses to arm 1, but if ESM3 sequence-only at 101 states lands at 0.52, that
same run is **+0.03 from structure** and is a real result. Without the
within-arm control you cannot tell those apart, and the whole arm is wasted.

Those controls cost 8 runs each and their embeddings already exist on disk.

## Steps

### Step 0 — per-mechanism rescoring · DONE, no GPU

Convertase 0.7073, zymogen 0.6813, unassigned 0.5503 at ±3, against a within-class
sd of 0.021–0.040. At ±1 the order reverses and convertase becomes the worst
class at 0.3791. The pooled number describes no subgroup.

**Keep this whatever else happens.** It is the paper's spine if every structure
arm fails.

### Step 1 — the baseline · 8 runs · IN FLIGHT

```bash
bash run_grammar_ablation.sh 10 100 results/esm2_rep1/config.json
```

Skips the four already done and fills to ten.

**Gate.** None. This *is* the baseline; there is no result that stops the study.
But record it against the 51-state 0.6153, because a large drop is itself a
finding about duration-coded CRFs.

### Step 2 — structures and 3Di · CPU only, run alongside step 1

```bash
pip install mini3di biotite
python -m src.utils.fetch_afdb_structures --out_dir structures/
python -m src.utils.make_3di --structures_dir structures/ --out_dir three_di/
```

**Gate.** `make_3di` must report ≥80% `ok`. Near zero means the manifest is wrong
and every downstream arm is a sequence-only run wearing a structure arm's name.

### Step 3 — the two sequence-only controls · 16 runs

ESM3 and ProstT5, sequence only, at 101 states. Embeddings already exist.

**Gate.** None. These are controls; they cannot fail, only inform.

### Step 4 — the structure arms · 16 runs

```bash
python -m src.utils.make_embeddings_esm3_struct \
    --structures_dir structures/ --out_dir embeddings/esm3_struct \
    --gpu_max_len 2000 --max_struct_len 1024

python -m src.utils.make_embeddings_prost5_struct \
    data/protein_sequences.fasta embeddings/prost5_fused \
    --tracks aa+3di --fuse renorm --three_di three_di/three_di.json
```

`--fuse renorm` keeps ProstT5 at 1024 dims so its head matches its own control
exactly. ESM3 is 1536 whatever tracks it is fed, so it needs no such choice.

8 replicates each.

**Gate, and this is the decision point.**

- **either arm beats arm 1** → the headline. Go to step 5.
- **neither beats arm 1, but one beats its own sequence-only control** → structure
  helps, this representation does not win. Publishable, and the honest framing.
  Go to step 5.
- **neither beats arm 1 nor its own control** → structure does not move this task.
  **Stop.** The paper is step 0 plus a well-controlled negative. Skip to step 6.

### Step 5 — scrambled controls and per-mechanism breakdown · 8 runs + no GPU

Only for whichever arm moved.

```bash
python -m src.utils.make_embeddings_esm3_struct ... --scramble_structure
python -m src.utils.make_embeddings_prost5_struct ... --shuffle_3di
python -m src.utils.score_by_mechanism results/<arm>_rep*/test_outputs.pickle --end_state 100
```

Beating the sequence-only control says the extra track helped. Beating the
**scrambled** control says *structure* helped. Only the second supports a
mechanistic claim.

Prediction from `PAPER.md`: least gain on convertase, most on zymogen prodomains.

### Step 6 — write up whichever is true

## Run budget

| step | runs | GPU |
|---|---|---|
| 1 baseline | 8 | in flight |
| 3 seq-only controls | 16 | yes |
| 4 structure arms | 16 | yes |
| 5 scrambled controls | 8 | only for the arm that moved |

48 runs. Steps 0, 2 and the analyses need no GPU.

## Deliberately excluded

- **ESM-2 + 3Di.** It is the likeliest arm to beat 0.6153, needing +0.035 where
  ESM3 and ProstT5 need +0.127 and +0.131. It is excluded because the chosen
  design is three clean representation arms, not a hybrid. `make_hybrid_embeddings.py`
  is on this branch if that decision is revisited.
- SaProt, the 2..100 dataset rebuild, equal-budget Optuna. Later work.
