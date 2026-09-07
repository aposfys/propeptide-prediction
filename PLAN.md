# The plan

One document. Where it disagrees with anything else in this repository, this one
wins. Steps run in order, each has a gate, and a failed gate has a written
consequence so the plan does not get renegotiated mid-run.

## The goal, stated once

**Beat 0.6153 on this benchmark.** That is ESM-2 at T4, sd 0.0242, n=10
(`esm2_rep*`), the best well-estimated arm in the study. Not `esm2_prop_final`
at 0.6262, which is a single run sitting inside that group's band.

At 8 replicates a resolvable win needs about **0.650**.

Metric throughout: segment-level propeptide F1 at ±3, micro-averaged on cluster
4. ±1 is recorded on every run and reported alongside, never selected on.

## Settled. Do not reopen.

| question | answer |
|---|---|
| CRF grammar | 51 states now; 101 (5..100) if step 1 clears. Never 151 |
| architecture | unchanged; `test_architecture.py` enforces it |
| min length | stays at 5 — below it is CAAX, a different reaction |
| where structure goes first | ESM-2, the arm that is already winning |
| dataset rebuild | separate project, after this one |
| tolerances | both, ±3 selects |

## Step 0 — per-mechanism rescoring · DONE

`score_by_mechanism.py` over the 10 ESM-2 replicates. No GPU.

Result: convertase 0.7073, zymogen 0.6813, unassigned 0.5503 at ±3. Spread 0.157
against a within-class sd of 0.021–0.040. At ±1 the ranking **reverses** —
convertase falls to 0.3791, the worst class. The pooled 0.6153 describes no
subgroup.

**This is a result the paper keeps regardless of what follows.** It is the
fallback spine if every structure arm fails.

## Step 1 — grammar ablation · RUNNING

4 replicates of ESM-2 at 101 states on unchanged data, where states 51–100 can
never be visited by a label.

```bash
python -m src.utils.summarize_results results/
```

**Gate.** Compare the `esm2_g101_rep*` group against `esm2_rep*` at 0.6153.

- within ~0.02 → the wider grammar is free. Proceed, and the paper may use 101
  states later.
- clearly lower → the enlarged transition matrix costs accuracy by itself. Stay
  at 51 states for everything below, and record it as a finding: a duration-coded
  CRF does not scale for free.

Either way **step 2 proceeds**. The grammar question does not block the structure
question.

## Step 2 — build the structural channel · CPU only

```bash
pip install mini3di biotite
python -m src.utils.fetch_afdb_structures --out_dir structures/
python -m src.utils.make_3di --structures_dir structures/ --out_dir three_di/
```

~8,000 downloads, so hours, but no GPU. Then:

```bash
python -m src.utils.make_hybrid_embeddings \
    --base_dir /mnt/storage/fysekidis/embeddings/esm2 \
    --three_di three_di/three_di.json --out_dir embeddings/esm2_3di --mode onehot
python -m src.utils.make_hybrid_embeddings \
    --base_dir /mnt/storage/fysekidis/embeddings/esm2 \
    --three_di three_di/three_di.json --out_dir embeddings/esm2_3di_shuf \
    --mode onehot --shuffle
```

**Gate.** `make_3di` must report ≥80% `ok`. Near zero means the manifest is
wrong and everything downstream is a sequence-only run wearing a structure arm's
name. Stop and fix.

## Step 3 — ESM-2 + 3Di, the main bet · 16 runs

1300 dims: ESM-2's 1280 plus 20 one-hot structural. Head grows 1.6%, so a gain
cannot be dismissed as capacity.

Replay the T4 config and change only the embeddings, exactly as the grammar
ablation does:

```bash
bash run_grammar_ablation.sh 8 50 results/esm2_rep1/config.json
```
with `PREFIX=esm2_3di_rep` and the embeddings directory overridden — or write the
8 commands from `results/esm2_rep1/config.json` with `--embeddings_dir
embeddings/esm2_3di --embedding_dim 1300`. Then 8 more against
`embeddings/esm2_3di_shuf`.

**Gate, and this is the one that decides the paper.**

- **≥ 0.650 and beats the shuffled control** → the headline. Go to step 4.
- **≥ 0.650 but the shuffled control matches it** → the gain is the extra
  20 dimensions, not structure. Report it as such. Go to step 5.
- **< 0.650** → structure does not clear the bar on ESM-2. **Stop building
  structure arms.** The paper becomes the mechanism-decomposition paper from step
  0 plus a documented negative result, which is publishable and honest. Skip to
  step 6.

## Step 4 — the richer channel · 16 runs, only if step 3 cleared

`--mode prostt5` gives 2304 dims. Confounded with capacity, which is why it is
second and not first, and why it needs its own shuffled control.

**Gate.** Beats the 1300-dim arm → report both and prefer the small one for the
claim. Does not → the 20-dim alphabet was enough, which is a cleaner result.

## Step 5 — per-mechanism breakdown · no GPU

```bash
python -m src.utils.score_by_mechanism results/esm2_3di_rep*/test_outputs.pickle
```

The prediction, from `PAPER.md`: structure helps least on convertase sites and
most on zymogen prodomains. If that ordering appears it is mechanistically
interpretable and it is the figure. If the gain is flat across mechanisms, say so
— a uniform gain is a weaker but still real result.

## Step 6 — write up whatever is true

Two outcomes are already publishable:

1. Structure clears the bar. Headline is the model; step 0 explains where the
   gain lives.
2. Structure does not. Headline is step 0 — `PROPEP` pools three mechanisms, the
   pooled metric describes no subgroup, the ranking reverses with tolerance —
   with the structure arms as a well-controlled negative.

Only then consider the dataset rebuild in `PAPER.md`. It is a second paper.

## What is deliberately NOT in this plan

- ESM3 and ProstT5 structure arms. They need +0.127 and +0.131 to clear the bar,
  larger than any effect measured in this study. They are ablations for a later
  paper, not the headline.
- SaProt. Worth adding, but it is a new dependency and a new arm; it belongs
  after step 3 answers whether structure moves this task at all.
- The 2..100 dataset rebuild. Separate project.
- Equal-budget Optuna for ESM3 and ProstT5. That is the thesis's next step, not
  the paper's.

## Run budget

| step | runs | GPU |
|---|---|---|
| 0 | 0 | no |
| 1 | 4 | yes, in flight |
| 2 | 0 | no |
| 3 | 16 | yes |
| 4 | 16 | only if step 3 cleared |
| 5 | 0 | no |

Worst case 36 GPU runs. Step 3 is the one that matters.
