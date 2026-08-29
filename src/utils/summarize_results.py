#!/usr/bin/env python3
'''Summarise every finished run under a results/ tree, from its own JSON files.

Reads config.json, test_metrics.json and valid_metrics.json and nothing else.
No number here is transcribed from a log, a notebook or a chat message; if a
figure is not derivable from those three files it does not appear in the output.

Three things it is built to answer, in order of importance:

  1. Is the comparison matched? The MATCHED-SET AUDIT crosses every distinct
     hyperparameter tuple against every embedder and prints the cell counts.
     An arm that only exists in one cell cannot be compared to an arm that only
     exists in another, and that is invisible when runs are listed by name.

  2. Are the "replicates" actually replicates? Runs are grouped by their full
     recorded config minus out_dir and seed, so two runs share a group only if
     every other argparse field is identical. The seed set of each group is
     printed: a group whose replicates all share one seed is not measuring seed
     variance, it is measuring GPU nondeterminism, and it must be captioned as
     such.

  3. How large is the noise floor? The single-run prediction band from the
     largest replicate group bounds what a one-run-per-arm comparison can
     resolve. Differences smaller than that band are not findings.

Statistics are Welch's unequal-variance t-test with Welch-Satterthwaite degrees
of freedom. The t distribution is evaluated through a regularised incomplete
beta so the script has no dependency beyond the standard library -- it has to
run wherever the results happen to live, which is not always where scipy is.
'''

import argparse
import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

# Bookkeeping fields, not experimental conditions. out_dir is the run's own
# name; seed is handled separately because whether it varies is the question,
# not part of the grouping.
NOT_A_CONDITION = ('out_dir', 'seed', 'outer_fold')

METRIC_KEYS = ('f1 propeptides', 'precision propeptides', 'recall propeptides')


# ---------------------------------------------------------------------------
# t distribution, via the regularised incomplete beta (Numerical Recipes 6.4)
# ---------------------------------------------------------------------------

def _betacf(a: float, b: float, x: float) -> float:
    maxit, eps, fpmin = 300, 3.0e-16, 1.0e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, maxit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbeta) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta) * _betacf(b, a, 1.0 - x) / b


def t_sf_two_sided(t: float, df: float) -> float:
    '''Two-sided p-value for Student's t.'''
    if df <= 0:
        return float('nan')
    return _betai(0.5 * df, 0.5, df / (df + t * t))


def t_crit(df: float, alpha: float = 0.05) -> float:
    '''Two-sided critical value, by bisection on the survival function.'''
    lo, hi = 0.0, 1000.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_sf_two_sided(mid, df) > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------

def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def sample_sd(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return float('nan')
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def welch(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    '''Welch's t-test. Returns nan fields rather than raising on n < 2.'''
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return {'diff': mean(a) - mean(b), 't': float('nan'),
                'df': float('nan'), 'p': float('nan')}
    va, vb = sample_sd(a) ** 2, sample_sd(b) ** 2
    se2 = va / na + vb / nb
    if se2 <= 0:
        return {'diff': mean(a) - mean(b), 't': float('inf'),
                'df': float('nan'), 'p': 0.0}
    se = math.sqrt(se2)
    t = (mean(a) - mean(b)) / se
    df = se2 ** 2 / (va ** 2 / (na ** 2 * (na - 1)) + vb ** 2 / (nb ** 2 * (nb - 1)))
    return {'diff': mean(a) - mean(b), 't': t, 'df': df,
            'p': t_sf_two_sided(abs(t), df)}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

class Run(object):
    def __init__(self, path: str, cfg: Dict, test: Optional[Dict],
                 valid: Optional[Dict]):
        self.path = path
        self.name = os.path.basename(path.rstrip('/'))
        self.cfg = cfg
        self.test = test
        self.valid = valid

    @property
    def f1(self) -> Optional[float]:
        return None if self.test is None else self.test.get('f1 propeptides')

    @property
    def embedder(self) -> str:
        d = self.cfg.get('embeddings_dir')
        if not d:
            # finetune.py runs carry no embeddings_dir: the backbone is in the
            # graph, not on disk. They are their own arm, never poolable with a
            # frozen arm.
            return '(finetuned backbone)'
        return os.path.basename(str(d).rstrip('/'))


def load_runs(root: str) -> Tuple[List[Run], List[str]]:
    runs, unfinished = [], []
    for dirpath, _dirnames, filenames in os.walk(root):
        if 'config.json' not in filenames:
            continue
        cfg = json.load(open(os.path.join(dirpath, 'config.json')))
        test = valid = None
        if 'test_metrics.json' in filenames:
            test = json.load(open(os.path.join(dirpath, 'test_metrics.json')))
        if 'valid_metrics.json' in filenames:
            valid = json.load(open(os.path.join(dirpath, 'valid_metrics.json')))
        if test is None or test.get('f1 propeptides') is None:
            unfinished.append(os.path.relpath(dirpath, root))
            continue
        runs.append(Run(dirpath, cfg, test, valid))
    return runs, sorted(unfinished)


def condition_key(cfg: Dict, all_keys: Sequence[str]) -> Tuple:
    '''Full config minus bookkeeping, normalised over the union of all keys.

    Normalising over the union matters because different branches wrote
    different argparse schemas; without it, a field one branch never had would
    split otherwise-identical runs into separate groups.
    '''
    return tuple((k, json.dumps(cfg.get(k), sort_keys=True))
                 for k in all_keys if k not in NOT_A_CONDITION)


def label_for(runs: Sequence[Run]) -> str:
    names = sorted(r.name for r in runs)
    if len(names) == 1:
        return names[0]
    prefix = os.path.commonprefix(names).rstrip('_-0123456789')
    return (prefix + '*') if len(prefix) >= 3 else ' + '.join(names)


def disambiguate(stats: List[Dict]) -> None:
    '''Make group labels unique in place.

    Two groups differ by construction -- they are different config tuples -- but
    their directory names need not say how, and a table with two identically
    named rows is worse than no table. Any collision gets suffixed with the
    fields that actually separate the colliding groups.
    '''
    by_label = {}
    for s in stats:
        by_label.setdefault(s['label'], []).append(s)
    for label, clash in by_label.items():
        if len(clash) < 2:
            continue
        keys = {k for s in clash for k in s['config']}
        differing = sorted(k for k in keys
                           if len({s['config'].get(k) for s in clash}) > 1)
        for s in clash:
            suffix = ' '.join('{}={}'.format(k, json.loads(s['config'][k]))
                              for k in differing if k in s['config'])
            s['label'] = (label + ' [' + suffix + ']') if suffix else label


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def fmt(x: Optional[float], nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return '   n/a'
    return ('{:.' + str(nd) + 'f}').format(x)


def report(root: str, min_n: int, dump: Optional[str]) -> None:
    runs, unfinished = load_runs(root)
    if not runs:
        print('No finished runs under ' + root)
        return

    all_keys = sorted({k for r in runs for k in r.cfg})

    groups = {}
    for r in runs:
        groups.setdefault(condition_key(r.cfg, all_keys), []).append(r)

    # ---- which config fields actually vary across groups -------------------
    varying = []
    for k in all_keys:
        if k in NOT_A_CONDITION:
            continue
        seen = {json.dumps(r.cfg.get(k), sort_keys=True) for r in runs}
        if len(seen) > 1:
            varying.append(k)

    print('=' * 78)
    print('RUNS: {} finished, {} with config but no test_metrics'.format(
        len(runs), len(unfinished)))
    if unfinished:
        for u in unfinished:
            print('    unfinished: ' + u)
    print('CONFIG FIELDS THAT VARY ACROSS RUNS: ' + (', '.join(varying) or '(none)'))
    novalid = [r.name for r in runs if r.valid is None]
    print('RUNS WITH NO valid_metrics.json: {}/{}'.format(len(novalid), len(runs)))
    if novalid:
        print('    ' + ', '.join(sorted(novalid)))

    # ---- matched-set audit -------------------------------------------------
    hp_fields = [k for k in ('lr', 'lora_lr', 'batch_size', 'dropout',
                             'weight_decay', 'epochs', 'patience', 'use_focal')
                 if k in all_keys]
    cells = {}
    embedders = set()
    for r in runs:
        hp = tuple(r.cfg.get(k) for k in hp_fields)
        embedders.add(r.embedder)
        cells.setdefault(hp, {}).setdefault(r.embedder, []).append(r.f1)

    print()
    print('=' * 78)
    print('MATCHED-SET AUDIT  (n and mean test F1 per hyperparameter x embedder)')
    print('Fields: ' + ', '.join(hp_fields))
    print('A row with one populated cell is an arm with no comparator.')
    print('-' * 78)
    emb_order = sorted(embedders)
    for hp in sorted(cells, key=lambda h: [str(v) for v in h]):
        populated = cells[hp]
        flag = '  <-- UNMATCHED' if len(populated) == 1 else ''
        print('  ' + ', '.join('{}={}'.format(k, v)
                               for k, v in zip(hp_fields, hp)) + flag)
        for e in emb_order:
            if e in populated:
                vals = populated[e]
                print('      {:28s} n={:<3d} mean {}'.format(
                    e, len(vals), fmt(mean(vals))))

    # ---- per-group statistics ---------------------------------------------
    stats = []
    for key, grp in groups.items():
        f1s = [r.f1 for r in grp]
        seeds = sorted({r.cfg.get('seed') for r in grp}, key=lambda s: str(s))
        stats.append({
            'label': label_for(grp),
            'runs': sorted(r.name for r in grp),
            'embedder': grp[0].embedder,
            'n': len(grp),
            'mean_f1': mean(f1s),
            'sd_f1': sample_sd(f1s),
            'min_f1': min(f1s),
            'max_f1': max(f1s),
            'spread': max(f1s) - min(f1s),
            'mean_precision': mean([r.test['precision propeptides'] for r in grp]),
            'mean_recall': mean([r.test['recall propeptides'] for r in grp]),
            'valid_f1s': [r.valid['f1 propeptides'] for r in grp
                          if r.valid is not None
                          and r.valid.get('f1 propeptides') is not None],
            'seeds': seeds,
            'distinct_seeds': len(seeds),
            'config': dict(key),
            'f1s': f1s,
        })
    stats.sort(key=lambda s: -s['mean_f1'])
    disambiguate(stats)

    print()
    print('=' * 78)
    print('GROUPS  (identical config apart from out_dir and seed)')
    print('-' * 78)
    print('{:26s} {:>3s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s}  {}'.format(
        'group', 'n', 'meanF1', 'sd', 'min', 'max', 'meanVal', 'seeds'))
    for s in stats:
        seed_note = ','.join(str(x) for x in s['seeds'])
        if s['n'] > 1 and s['distinct_seeds'] == 1:
            seed_note += '  (ALL ONE SEED)'
        val = mean(s['valid_f1s']) if s['valid_f1s'] else None
        if s['valid_f1s'] and len(s['valid_f1s']) < s['n']:
            seed_note += '  [val on {}/{}]'.format(len(s['valid_f1s']), s['n'])
        print('{:26s} {:>3d} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s}  {}'.format(
            s['label'][:26], s['n'], fmt(s['mean_f1']), fmt(s['sd_f1']),
            fmt(s['min_f1']), fmt(s['max_f1']), fmt(val), seed_note))

    # ---- noise floor -------------------------------------------------------
    replicated = [s for s in stats if s['n'] >= min_n]
    print()
    print('=' * 78)
    print('NOISE FLOOR  (95% prediction band for one further run of the same config)')
    print('mean +/- t(n-1,.975) * sd * sqrt(1 + 1/n)')
    print('-' * 78)
    for s in replicated:
        half = t_crit(s['n'] - 1) * s['sd_f1'] * math.sqrt(1.0 + 1.0 / s['n'])
        s['pred_half_width'] = half
        print('  {:26s} n={:<3d} {} +/- {}   [{}, {}]'.format(
            s['label'][:26], s['n'], fmt(s['mean_f1']), fmt(half),
            fmt(s['mean_f1'] - half), fmt(s['mean_f1'] + half)))
        if s['distinct_seeds'] == 1:
            print('      every run used seed {} -- this band is run-to-run '
                  'nondeterminism, not seed variance'.format(s['seeds'][0]))

    # ---- pairwise Welch ----------------------------------------------------
    print()
    print('=' * 78)
    print("PAIRWISE WELCH t-TEST  (groups with n >= {})".format(min_n))
    print('-' * 78)
    comparisons = []
    for i in range(len(replicated)):
        for j in range(i + 1, len(replicated)):
            a, b = replicated[i], replicated[j]
            w = welch(a['f1s'], b['f1s'])
            comparisons.append({'a': a['label'], 'b': b['label'],
                                'n_a': a['n'], 'n_b': b['n'], **w})
            print('  {:24s} vs {:24s}  diff {}  t={:6.3f}  df={:5.2f}  p={:.4g}'
                  .format(a['label'][:24], b['label'][:24], fmt(w['diff']),
                          w['t'], w['df'], w['p']))
    if not comparisons:
        print('  (fewer than two groups reach n >= {})'.format(min_n))

    # ---- singletons --------------------------------------------------------
    singles = [s for s in stats if s['n'] == 1]
    if singles and replicated:
        # Reference the group with the most runs, not the widest band: the
        # widest band belongs to whichever group happened to contain a collapsed
        # run, and quoting it would overstate the noise floor. The full range is
        # printed alongside so the choice is visible rather than assumed.
        ref = max(replicated, key=lambda s: s['n'])
        bands = [s['pred_half_width'] for s in replicated]
        print()
        print('=' * 78)
        print('SINGLE-RUN CONFIGS  ({} of them), sorted by F1'.format(len(singles)))
        print('Reference band: +/- {} (from {}, the largest replicate group, '
              'n={}).'.format(fmt(ref['pred_half_width']), ref['label'], ref['n']))
        print('Across all replicate groups the band ranges +/- {} to +/- {}.'
              .format(fmt(min(bands)), fmt(max(bands))))
        print('Two entries closer than the band are not distinguishable here.')
        print('-' * 78)
        for s in singles:
            print('  {:34s} {}  P {}  R {}'.format(
                s['label'][:34], fmt(s['mean_f1']),
                fmt(s['mean_precision']), fmt(s['mean_recall'])))

    if dump:
        payload = {'root': os.path.abspath(root),
                   'n_runs': len(runs),
                   'unfinished': unfinished,
                   'varying_config_fields': varying,
                   'groups': [{k: v for k, v in s.items()} for s in stats],
                   'comparisons': comparisons}
        json.dump(payload, open(dump, 'w'), indent=2, default=str)
        print()
        print('Wrote ' + dump)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', nargs='?', default='results',
                   help='directory to walk (default: results)')
    p.add_argument('--min_n', type=int, default=3,
                   help='group size at which a group counts as replicated')
    p.add_argument('--json', dest='dump', default=None,
                   help='also write the full summary to this JSON path')
    a = p.parse_args()
    report(a.root, a.min_n, a.dump)


if __name__ == '__main__':
    main()
