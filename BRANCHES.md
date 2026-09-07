# Branches

One branch per embedding model. Each is a full working copy, not a patch on top of another.

| branch | embeddings | what it is |
|---|---|---|
| `main` | ESM-2, 1280 | propeptides only, the reference arm |
| `baseline-upstream` | ESM-2, 1280 | faithful to upstream: joint peptides + propeptides, upstream metric |
| `esm3-propeptide` | ESM3 `esm3_sm_open_v1`, 1536 | propeptides only. Also holds the structure channel, LoRA fine-tuning, Optuna/nested CV, and the analysis scripts |
| `esm3-full` | ESM3, 1536 | joint peptides + propeptides |
| `esm3-multimodal` | ESM3, 1536 | `esm3-propeptide` with the structure conditioning made answerable: the two structural pathways separable, a scrambled-structure control, and a powered design. Nothing run yet |
| `prost5-propeptide` | ProstT5, 1024 | propeptides only |
| `prost5-full` | ProstT5, 1024 | joint peptides + propeptides |
| `prost5-multimodal` | ProstT5, 1024 or 2048 | ProstT5 over amino acids **and** Foldseek 3Di, concatenated, with a shuffled-3Di control. Nothing run yet |
| `archive/eirini-esm1b` | ESM-1b | an older contributed fork, kept for the record. Different code and a different metric version, so its numbers don't belong in a table with the rest |

The two `*-multimodal` branches are the structure arms: each takes its
sequence-only parent and adds input the parent did not have, holding the head,
the data, the splits, the metric and the training budget fixed. Both carry a
negative control, because in both cases "the extra track helped" and "structure
helped" are different claims. Neither has been run — they carry the pipeline and
the design, not results. See each branch's `NEXT_STEPS.md`.

They also share three changes that are not about structure: both boundary
tolerances (±1 and ±3) are scored on every run, the CRF grammar size is
configurable rather than four hardcoded 51s, and `GRAMMAR.md` measures what the
published 5..50 length window actually covers in UniProt.

Don't compare F1 across branches without checking which metric produced it.
`baseline-upstream` keeps the upstream metric on purpose, so that it reproduces the
published figures. Every other branch fixes that metric, which shifts the values.
