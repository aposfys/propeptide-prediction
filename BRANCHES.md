# Branches

One branch per embedding model. Each is a full working copy, not a patch on top of another.

| branch | embeddings | what it is |
|---|---|---|
| `main` | ESM-2, 1280 | propeptides only, the reference arm |
| `baseline-upstream` | ESM-2, 1280 | faithful to upstream: joint peptides + propeptides, upstream metric |
| `esm3-propeptide` | ESM3 `esm3_sm_open_v1`, 1536 | propeptides only. Also holds the structure channel, LoRA fine-tuning, Optuna/nested CV, and the analysis scripts |
| `esm3-full` | ESM3, 1536 | joint peptides + propeptides |
| `prost5-propeptide` | ProstT5, 1024 | propeptides only |
| `prost5-full` | ProstT5, 1024 | joint peptides + propeptides |
| `archive/eirini-esm1b` | ESM-1b | an older contributed fork, kept for the record. Different code and a different metric version, so its numbers don't belong in a table with the rest |

Don't compare F1 across branches without checking which metric produced it.
`baseline-upstream` keeps the upstream metric on purpose, so that it reproduces the
published figures. Every other branch fixes that metric, which shifts the values.
