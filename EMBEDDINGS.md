# ESM3 embedding scaling

Why every ESM3 result predating 2026-08-19 is invalid, and how to repair embeddings
without re-running ESM3.

`src/utils/make_embeddings.py` used to save `ESMOutput.embeddings`. That field is
ESM3's **raw pre-LayerNorm residual stream**, not the representation its own
output heads consume — `TransformerStack.forward` returns `self.norm(x), x, …`
and `ESM3.forward` unpacks it as `x, embedding, _`. fair-esm does the opposite for
ESM-2 (it applies `emb_layer_norm_after` and overwrites `representations[33]`), so
**the ESM-2 baseline trained on normalised features and ESM3 did not.**

| features | per-token ‖x‖ |
|---|---|
| ESM-2 L33 (the baseline) | 10.12 |
| ESM3 after `transformer.norm` | 11.62 |
| ESM3 `.embeddings`, as previously saved | **9792.73** |

~840× too large. `LSTMCNN` has no input normalisation, so this saturates **90.7%**
of the biLSTM gates at initialisation. It is why the optimal learning rate
collapsed, why ESM-2's T4 settings "broke" ESM3, and why the first 30-trial search
sat on a flat plateau. Every ESM3 result predating the fix is invalid — see
[RESULTS.md](RESULTS.md).

**Old embeddings can be repaired without re-running ESM3.** The final norm is
per-token, so it commutes with the BOS/EOS slice:

```bash
python -m src.utils.renorm_esm3_embeddings /path/to/esm3 /path/to/esm3_normed
```

`preflight.py` now refuses to start a run on mis-scaled embeddings.