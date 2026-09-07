'''
Assert the training core is byte-identical to the other two arm branches.

The three arms are separate branches, by the convention in BRANCHES.md. That is
only safe if the code that TRAINS is the same on all three -- otherwise a
difference in F1 could be the training loop rather than the representation, and
no amount of replication separates those.

This is not hypothetical. Before the arms were split, `esm3-multimodal` and
`prost5-multimodal` differed in train_loop_crf.py, crf_models.py, dataset.py AND
manuscript_metrics.py, and RESULTS.md already warns against comparing F1 across
branches for exactly this reason.

    python test_core_identical.py

Needs a git remote. Compares blob hashes, so it detects a one-character change.
'''
import subprocess
import sys

CORE = [
    'src/train_loop_crf.py',
    'src/models/crf_models.py',
    'src/models/lstm_cnn.py',
    'src/models/multi_tag_crf.py',
    'src/utils/dataset.py',
    'src/utils/manuscript_metrics.py',
    'src/utils/crf_label_utils.py',
    'run.py',
]
BRANCHES = ['esm2-101', 'esm3-101', 'prost5-101']

failures = []


def blob(ref, path):
    try:
        return subprocess.check_output(['git', 'rev-parse', f'{ref}:{path}'],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except subprocess.CalledProcessError:
        return None


def main():
    subprocess.run(['git', 'fetch', '--quiet', 'origin'], check=False)
    print('Comparing the training core across the three arm branches\n')
    for path in CORE:
        hashes = {}
        for branch in BRANCHES:
            for ref in (f'origin/{branch}', branch):
                digest = blob(ref, path)
                if digest:
                    hashes[branch] = digest
                    break
        if len(hashes) < len(BRANCHES):
            missing = [b for b in BRANCHES if b not in hashes]
            print(f'  SKIP  {path}  (not found on {", ".join(missing)})')
            continue
        identical = len(set(hashes.values())) == 1
        print(('  PASS  ' if identical else '  FAIL  ') + path)
        if not identical:
            for branch, digest in hashes.items():
                print(f'          {branch:14} {digest[:12]}')
            failures.append(path)

    print()
    print(f'{len(failures)} file(s) differ')
    if failures:
        print('A difference here means a cross-arm F1 comparison is confounded '
              'with the training code.')
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
