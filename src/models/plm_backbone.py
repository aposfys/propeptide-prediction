'''
Trainable ESM3 backbone for DeepPeptide.

Everything in this repo so far has consumed *precomputed, frozen* embeddings.
This module makes the encoder part of the graph, so gradients from the CRF
reach the transformer. Five ESM3-specific facts are baked in, each verified
against `esm==3.2.1` rather than assumed:

1. `ESMOutput.embeddings` is the PRE-LayerNorm residual stream.
   `TransformerStack.forward` returns `self.norm(x), x, hiddens` and
   `ESM3.forward` unpacks it as `x, embedding, _` before calling
   `self.output_heads(x, embedding)`. Replacing output_heads returns the
   post-norm tensor by construction, so the ~840x scale trap that invalidated
   the earlier runs cannot recur, and six unused generative RegressionHeads
   leave the graph.

2. Geometric attention in block 0 contributes EXACTLY ZERO for sequence-only
   input, and disabling it is bit-exact. With `structure_coords=None`, ESM3
   fills NaN coordinates, `build_affine3d_from_coordinates` returns an
   all-False `affine_mask`, and `mask_and_zero_frameless=True` zeroes
   `attn_out`; `out_proj` has no bias, so the block contributes 0.
   `UnifiedTransformerBlock.forward` gates it behind `if self.use_geom_attn:`,
   so the flag genuinely skips the work. What it costs if left on is memory:
   three B x heads x L x L tensors, i.e. 3.0 GiB at L=1024 and 37.1 GiB at
   L=3600 in fp32 -- the 37.6 GiB OOM already recorded in RESULTS.md.

3. LoRA targets are `attn.layernorm_qkv.1`, `attn.out_proj`, `ffn.1`, `ffn.3`.
   `layernorm_qkv` and `ffn` are `nn.Sequential`, NOT Linear, so a bare
   `target_modules=['layernorm_qkv']` (the ESM-2 convention) matches a
   container and attaches nothing.

4. The sequence tokenizer is a HuggingFace tokenizer returning `input_ids` and
   `attention_mask` -- NOT `sequence_tokens`, which is the name of ESM3's
   forward argument.

5. It RIGHT-pads, so a batch row reads [BOS, r1..rn, EOS, PAD, ...]. A naive
   `attention_mask[:, 1:-1]` residue mask therefore marks the EOS slot as a
   valid residue for every sequence shorter than the batch maximum, shifting
   one spurious background position into each. The mask is built from true
   sequence lengths instead.

No `peft` dependency: LoRA is ~40 lines and avoids a wrapper that would have to
forward ESM3's keyword-only signature.
'''
import math
import re
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# LoRA
# --------------------------------------------------------------------------

#: Per-block Linear modules worth adapting, with r=8 parameter cost at
#: d_model=1536 and expansion 8/3:
#:   attn.layernorm_qkv.1  (4608, 1536)   49,152
#:   attn.out_proj         (1536, 1536)   24,576
#:   ffn.1                 (8192, 1536)   77,824
#:   ffn.3                 (1536, 4096)   45,056
#: => 196,608 per block; 12 blocks = 2.36M params = 0.17% of ESM3's 1.4B.
LORA_TARGETS: Tuple[str, ...] = (
    'attn.layernorm_qkv.1',
    'attn.out_proj',
    'ffn.1',
    'ffn.3',
)


class LoRALinear(nn.Module):
    '''Frozen base Linear + trainable low-rank update, B initialised to zero.

    B=0 means the adapted model is *identical* to the pretrained one at step 0,
    so a LoRA run and a frozen run start from the same point and any difference
    is attributable to the adaptation rather than to the rewrite.
    '''

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        h = self.dropout(x)
        # .to(h.dtype) keeps the fp32 master parameters usable under bf16
        # autocast without storing a second copy.
        h = h @ self.lora_A.to(h.dtype).t()
        h = h @ self.lora_B.to(h.dtype).t()
        return out + self.scaling * h

    def extra_repr(self) -> str:
        return f'r={self.r}, scaling={self.scaling:.3g}'


def inject_lora(
    transformer: nn.Module,
    n_last_blocks: int,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    targets: Sequence[str] = LORA_TARGETS,
) -> int:
    '''Wrap `targets` in the last `n_last_blocks` blocks of a TransformerStack.

    Adapting only the top of the stack follows PEFT-SP (Zeng et al. 2024), which
    LoRA-adapts the final 25 of 33 ESM-2 layers for signal-peptide cleavage. It
    also matches this project's own layer sweep: the late blocks are where the
    propeptide signal lives (layer 48 > 44 > 30 on test F1).

    Returns the number of trainable LoRA parameters added.
    '''
    blocks = transformer.blocks
    if not 0 < n_last_blocks <= len(blocks):
        raise ValueError(f'n_last_blocks must be in 1..{len(blocks)}, got {n_last_blocks}')

    added = 0
    for block in blocks[len(blocks) - n_last_blocks:]:
        for path in targets:
            parent_path, _, attr = path.rpartition('.')
            parent = block.get_submodule(parent_path) if parent_path else block
            base = getattr(parent, attr)
            if not isinstance(base, nn.Linear):
                raise TypeError(
                    f"LoRA target '{path}' resolved to {type(base).__name__}, not nn.Linear. "
                    "In ESM3, 'layernorm_qkv' and 'ffn' are nn.Sequential -- index into "
                    "them ('layernorm_qkv.1', 'ffn.1', 'ffn.3')."
                )
            wrapped = LoRALinear(base, r=r, alpha=alpha, dropout=dropout)
            setattr(parent, attr, wrapped)
            added += wrapped.lora_A.numel() + wrapped.lora_B.numel()
    return added


# --------------------------------------------------------------------------
# Backbone
# --------------------------------------------------------------------------

class _ReturnPostNorm(nn.Module):
    '''Replaces ESM3.output_heads.

    ESM3.forward ends with `return self.output_heads(x, embedding)` where `x` is
    post-LayerNorm and `embedding` is pre-LayerNorm. Swapping the heads for this
    returns the post-norm tensor directly and drops the six generative
    RegressionHeads (sequence/structure/ss8/sasa/function/residue) from the
    graph -- parameters and activations this task never uses.
    '''

    def forward(self, x: torch.Tensor, embed: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        return x


def _checkpointed(fn, checkpoint):
    def wrapped(*args, **kwargs):
        if torch.is_grad_enabled() and any(
            isinstance(a, torch.Tensor) and a.requires_grad for a in args
        ):
            return checkpoint(fn, *args, use_reentrant=False, **kwargs)
        return fn(*args, **kwargs)
    return wrapped


class PLMBackbone(nn.Module):
    '''ESM3 encoder returning per-residue representations.

    forward(sequences) -> (reps, mask)
        reps : (B, L, d_model) float32, post-final-LayerNorm, BOS/EOS stripped
        mask : (B, L) float32, 1 for real residues

    Args:
        n_lora_blocks: adapt the last N blocks. 0 = fully frozen, which
                    reproduces the existing precomputed-embedding pipeline on
                    the fly and is the control every sweep should carry.
        max_len: crop longer sequences. 2,048 leaves ZERO unreachable
                    propeptides on valid and test; 1,024 would cost -0.0027 F1 and
                    would preferentially delete C-terminal propeptides of long
                    precursors, a biological class rather than a random slice.
        dtype: float32 by default and deliberately so. `ESM3.from_pretrained`
                    yields bfloat16 on CUDA and float32 on CPU, which silently
                    changes numerics between machines -- the trap documented in
                    RESULTS.md. Keeping master weights in fp32 and using bf16
                    autocast at the call site gets the speed without the
                    ambiguity, and keeps the frozen control comparable to the
                    cached-embedding runs.
    '''

    def __init__(
        self,
        model_name: str = 'esm3_sm_open_v1',
        n_lora_blocks: int = 12,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        grad_checkpoint: bool = True,
        max_len: int = 2048,
        device: torch.device | str = 'cuda',
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        from esm.models.esm3 import ESM3

        self.model_name = model_name
        self.max_len = max_len
        self.grad_checkpoint = grad_checkpoint

        self.plm = ESM3.from_pretrained(model_name, device=torch.device(device)).to(dtype)
        self.tokenizer = self.plm.tokenizers.sequence

        # (1) post-norm reps, and drop the unused generative heads
        self.plm.output_heads = _ReturnPostNorm()

        # (2) sequence-only => geometric attention is an exact no-op. Only
        #     block 0 carries it (n_layers_geom=1), but loop rather than
        #     hardcoding the index so a different stack config cannot leave one
        #     enabled and quietly reintroduce the O(L^2) allocation.
        self.n_geom_disabled = 0
        for block in self.plm.transformer.blocks:
            if getattr(block, 'use_geom_attn', False):
                block.use_geom_attn = False
                self.n_geom_disabled += 1

        self.transformer = self.plm.transformer
        self.d_model = self.transformer.norm.normalized_shape[0]

        for p in self.plm.parameters():
            p.requires_grad_(False)

        self.n_lora_blocks = n_lora_blocks
        self.n_lora_params = 0
        if n_lora_blocks > 0:
            self.n_lora_params = inject_lora(
                self.transformer, n_lora_blocks, lora_r, lora_alpha, lora_dropout
            )
        if grad_checkpoint:
            self._enable_grad_checkpointing()

    # -- gradient checkpointing ---------------------------------------------
    def _enable_grad_checkpointing(self) -> None:
        '''Recompute each block in backward. Trades ~30% step time for a large
        activation saving: without it, 48 blocks x B x L x 1536 activations are
        all retained.'''
        from torch.utils.checkpoint import checkpoint

        for block in self.transformer.blocks:
            if getattr(block, '_ckpt_wrapped', False):
                continue
            block.forward = _checkpointed(block.forward, checkpoint)
            block._ckpt_wrapped = True

    # -- forward ------------------------------------------------------------
    def tokenize(self, sequences: List[str], device: torch.device):
        seqs = [s[: self.max_len] for s in sequences]
        enc = self.tokenizer(seqs, padding=True, return_tensors='pt')
        # HuggingFace keys. 'sequence_tokens' is the name of ESM3.forward's
        # argument, not of anything the tokenizer emits.
        tokens = enc['input_ids'].to(device)
        attn = enc['attention_mask'].to(device)
        lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long, device=device)
        return tokens, attn, lengths

    def forward(self, sequences: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(p for p in self.parameters()).device
        tokens, attn, lengths = self.tokenize(sequences, device)

        # ESM3 attention treats `sequence_id` as a grouping key: positions with
        # equal ids attend to each other. Giving pad a different id stops real
        # residues from attending to padding.
        sequence_id = (~attn.bool()).long()

        out = self.plm(sequence_tokens=tokens, sequence_id=sequence_id)
        reps = out.float()[:, 1:-1, :]          # strip BOS / EOS positions

        # The tokenizer RIGHT-pads, so for any sequence shorter than the batch
        # maximum the [1:-1] slice still holds its EOS and some PADs. Deriving
        # the mask from attn[:, 1:-1] would mark that EOS as a valid residue and
        # feed one spurious background position per short sequence into the CRF.
        idx = torch.arange(reps.shape[1], device=device).unsqueeze(0)
        mask = (idx < lengths.unsqueeze(1)).float()
        return reps, mask


# --------------------------------------------------------------------------
# Optimiser groups
# --------------------------------------------------------------------------

def param_groups(
    model: nn.Module,
    head_lr: float,
    lora_lr: float,
    weight_decay: float = 1e-4,
) -> List[dict]:
    '''Discriminative learning rates.

    The head is randomly initialised and wants the large LR this project already
    found (5.5e-3). The LoRA adapters sit on a pretrained backbone and want
    1-2 orders of magnitude less; running them at head LR is the standard way to
    destroy a pretrained encoder in the first few hundred steps.

    CRF transition parameters are excluded from weight decay: forbidden
    transitions are pinned at -1e10, and decaying them is meaningless.
    '''
    lora, head_decay, head_no_decay = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'lora_' in name:
            lora.append(p)
        elif re.search(r'(transitions|bias|norm|LayerNorm)', name) or p.ndim == 1:
            head_no_decay.append(p)
        else:
            head_decay.append(p)
    groups = [
        {'params': head_decay, 'lr': head_lr, 'weight_decay': weight_decay, 'name': 'head'},
        {'params': head_no_decay, 'lr': head_lr, 'weight_decay': 0.0, 'name': 'head_no_decay'},
    ]
    # An empty group is legal but makes the frozen control's optimiser state
    # confusing to read; omit it instead.
    if lora:
        groups.insert(0, {'params': lora, 'lr': lora_lr,
                          'weight_decay': weight_decay, 'name': 'lora'})
    return groups


def summarise(model: nn.Module) -> str:
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in model.parameters())
    lora = sum(p.numel() for n, p in model.named_parameters() if 'lora_' in n and p.requires_grad)
    return (f'trainable {tr:,} / {tot:,} ({100 * tr / tot:.3f}%) '
            f'| lora {lora:,} | head {tr - lora:,}')
