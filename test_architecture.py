'''
Assert that the model this branch builds is the ORIGINAL model, unchanged.

The multimodal branches add input channels and make the CRF grammar
configurable. Neither is supposed to touch the architecture at its defaults, and
"supposed to" is not evidence. This pins the model down against reference values
measured from the parent branches (`esm3-propeptide` and `prost5-propeptide`)
before any of those changes existed.

    python test_architecture.py

Exits 0 if the model is identical to the original, 1 otherwise. No GPU, no
embeddings, no network.

If a reference value here ever needs updating, that is the point: it means the
architecture moved, and the reason has to be written down rather than absorbed.
'''
import argparse
import hashlib
import sys

import torch

from src.train_loop_crf import get_model

# Measured on the parent branches at their native embedding dimension.
# The CRF constraint mask is the grammar as the model actually sees it, so its
# hash is the strongest single check available without training anything.
REFERENCE = {
    1536: {'n_params': 241521, 'branch': 'esm3-propeptide'},
    1024: {'n_params': 192369, 'branch': 'prost5-propeptide'},
}
CRF_MASK_SHA = 'b0a6f98158b2104738aa700d3d796b68a724d81e21774cfdab4523a132e78da9'
CRF_MASK_SUM = 98.0
N_TRANSITIONS = 99          # 98 unique; upstream adds (49, 50) twice
N_TENSORS = 20

failures = []


def check(condition, message):
    print(('  PASS  ' if condition else '  FAIL  ') + message)
    if not condition:
        failures.append(message)


def build(embedding_dim, **overrides):
    args = dict(model='lstmcnncrf', embedding_dim=embedding_dim, dropout=0.1,
                conv_dropout=0.1, kernel_size=3, num_filters=32, hidden_size=64)
    args.update(overrides)
    return get_model(argparse.Namespace(**args))


print('--- the CRF is the original one ---')
model = build(1024)
check(model.num_states == 51, 'num_states is 51')
check(model.crf.num_tags == 51, 'the CRF has 51 tags')
check((model.max_len, model.min_len) == (50, 5), 'the grammar is 5..50')

transitions, starts, ends = type(model).get_crf_constraints(50, 5)
check(len(transitions) == N_TRANSITIONS,
      f'{N_TRANSITIONS} transitions (got {len(transitions)})')
check(starts == [0, 1] and ends == [0, 50], 'allowed starts/ends unchanged')

mask = model.crf._constraint_mask
digest = hashlib.sha256(mask.cpu().numpy().tobytes()).hexdigest()
check(digest == CRF_MASK_SHA,
      'the CRF constraint mask is byte-identical to the original')
check(float(mask.sum()) == CRF_MASK_SUM,
      f'the mask permits {int(CRF_MASK_SUM)} transitions')

print('--- the head is the original one ---')
state_dict = model.state_dict()
check(len(state_dict) == N_TENSORS, f'{N_TENSORS} tensors in the state dict')
for embedding_dim, reference in sorted(REFERENCE.items()):
    n = sum(p.numel() for p in build(embedding_dim).parameters() if p.requires_grad)
    check(n == reference['n_params'],
          f'{embedding_dim}-dim input -> {reference["n_params"]:,} trainable params, '
          f'as on {reference["branch"]} (got {n:,})')

print('--- passing the grammar flags at their defaults changes nothing ---')
explicit = build(1024, max_peptide_len=50, min_peptide_len=5)
check(sum(p.numel() for p in explicit.parameters() if p.requires_grad)
      == REFERENCE[1024]['n_params'],
      'explicit --max_peptide_len 50 --min_peptide_len 5 is the same model')
check(hashlib.sha256(explicit.crf._constraint_mask.cpu().numpy().tobytes()).hexdigest()
      == CRF_MASK_SHA,
      'and the same CRF constraint mask')

print('--- only a wider grammar changes it, and only where expected ---')
wide = build(1024, max_peptide_len=105, min_peptide_len=2)
check(wide.num_states == 106, 'max_peptide_len 105 gives 106 states')
wide_only = {k for k in wide.state_dict()
             if list(wide.state_dict()[k].shape) != list(state_dict[k].shape)}
check(wide_only == {'crf._constraint_end_mask', 'crf._constraint_mask',
                    'crf._constraint_start_mask', 'crf.end_transitions',
                    'crf.start_transitions', 'crf.transitions'},
      'a wider grammar resizes the CRF tensors and nothing else')

print('--- input width is the only thing an extra channel changes ---')
narrow, doubled = build(1024).state_dict(), build(2048).state_dict()
differing = {k for k in narrow if list(narrow[k].shape) != list(doubled[k].shape)}
check(differing == {'feature_extractor.conv1.weight'},
      'a 2048-dim input widens conv1 and nothing else')
check(doubled['feature_extractor.conv1.weight'].numel()
      - narrow['feature_extractor.conv1.weight'].numel() == 98304,
      'that costs exactly 98,304 parameters')

print()
print(f'{len(failures)} failure(s)')
sys.exit(1 if failures else 0)
