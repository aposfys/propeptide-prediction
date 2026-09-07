'''
Assert that the configurable grammar reproduces the published one at its defaults.

`--min_peptide_len` and `--max_peptide_len` exist so a follow-on study on
unfiltered UniProt does not have to start by editing four hardcoded constants
(see GRAMMAR.md). The risk of that change is that it silently alters the model
everyone's existing numbers came from. This is the test that says it did not.

    python test_grammar.py

Exits 0 if the defaults are byte-for-byte the old behaviour, 1 otherwise.
No GPU, no embeddings, no network.
'''
import sys

import numpy as np
import torch

from src.models.crf_models import CRFBaseModel, LSTMCNNCRF
from src.utils.crf_label_utils import peptide_list_to_label_sequence

failures = []


def check(condition, message):
    print(('  PASS  ' if condition else '  FAIL  ') + message)
    if not condition:
        failures.append(message)


print('--- the default grammar is the published one ---')

# The reference values below are the literals the code used before the grammar
# became configurable: max_len 50, min_len 5, 51 states.
transitions, starts, ends = CRFBaseModel.get_crf_constraints(50, 5)

# Recompute the transition list the way the code did before the grammar became
# configurable, and demand an exact match. Comparing against a remembered count
# would not catch a reordering or a swapped pair.
reference = [(0, 0), (0, 1), (50, 1), (50, 0)]
for i in range(1, 50):
    reference.append((i, i + 1))
    if i > 4:
        reference.append((3, i))
reference.append((49, 50))
check(transitions == reference,
      'the transition list is identical to the pre-configurable code '
      f'({len(reference)} entries, one of them a duplicate upstream already had)')
check(starts == [0, 1], 'allowed starts are [0, 1]')
check(ends == [0, 50], 'allowed ends are [0, 50]')

model = LSTMCNNCRF(input_size=64)
check(model.max_len == 50 and model.min_len == 5, 'LSTMCNNCRF defaults to 5..50')
check(model.num_states == 51, 'LSTMCNNCRF defaults to 51 states')
check(model.crf.num_tags == 51, 'the CRF itself has 51 tags')

print('--- label encoding is unchanged at the defaults ---')
# Spot-check the three shapes the encoder can produce: the minimum length, a
# middle length, and the maximum.
for length in (5, 11, 50):
    spans = [(10, 10 + length - 1)]
    encoded = peptide_list_to_label_sequence(spans, 200, start_state=1, max_len=50, min_len=5)
    segment = encoded[9:9 + length]
    expected = np.concatenate([np.arange(1, 4),
                               np.arange(49 - (length - 5), 51)])
    check(np.array_equal(segment, expected),
          f'length-{length} propeptide encodes to the published state path')
    check(segment.max() == 50 and segment.min() == 1,
          f'length-{length} path stays inside states 1..50')

print('--- widening the grammar does what it says ---')
wide = LSTMCNNCRF(input_size=64, num_states=106, max_len=105, min_len=2)
check(wide.num_states == 106 and wide.crf.num_tags == 106, '106-state grammar builds')
check(wide.max_len == 105 and wide.min_len == 2, 'wide grammar keeps 2..105')

encoded = peptide_list_to_label_sequence([(10, 87)], 200, start_state=1, max_len=105, min_len=2)
segment = encoded[9:9 + 78]
check(len(segment) == 78 and segment.max() <= 105,
      'a 78-residue propeptide is representable at max_len 105')
try:
    peptide_list_to_label_sequence([(10, 87)], 200, start_state=1, max_len=50, min_len=5)
    check(False, 'the 51-state grammar refuses a 78-residue propeptide')
except ValueError:
    check(True, 'the 51-state grammar refuses a 78-residue propeptide instead of '
                'emitting negative states behind a print statement')

try:
    peptide_list_to_label_sequence([(10, 12)], 200, start_state=1, max_len=50, min_len=5)
    check(False, 'the grammar refuses a propeptide below min_len')
except ValueError:
    check(True, 'the grammar refuses a 3-residue propeptide at min_len 5')

check(peptide_list_to_label_sequence([(10, 12)], 200, start_state=1,
                                     max_len=50, min_len=2)[9:12].min() > 0,
      'the same 3-residue propeptide is representable at min_len 2, '
      'with no extra states')

print('--- inconsistent grammars are refused, not silently accepted ---')
for kwargs, why in (
        (dict(num_states=51, max_len=105, min_len=5), 'num_states != max_len + 1'),
        (dict(num_states=106, max_len=105, min_len=0), 'min_len below 1'),
        (dict(num_states=106, max_len=105, min_len=200), 'min_len above max_len')):
    try:
        LSTMCNNCRF(input_size=64, **kwargs)
        check(False, f'rejects {why}')
    except ValueError:
        check(True, f'rejects {why}')

print('--- both tolerances are scored ---')
from src.train_loop_crf import TOLERANCES, flatten_tolerances
check(TOLERANCES == [1, 3], 'TOLERANCES is [1, 3]')
check(TOLERANCES[-1] == 3, 'the selection tolerance is still 3')
flat = flatten_tolerances([{'f1 propeptides': 0.1}, {'f1 propeptides': 0.2}])
check(flat == {'f1 propeptides@1': 0.1, 'f1 propeptides@3': 0.2},
      'metrics are suffixed per tolerance')

print()
print(f'{len(failures)} failure(s)')
sys.exit(1 if failures else 0)
