'''
Show how far a set of runs has got, reading only what is on disk.

    python src/utils/progress.py "results/esm2_rep*"
    watch -n60 'python src/utils/progress.py "results/*"'

Neither run.py nor finetune.py writes a log file -- the per-epoch lines go to
stdout, so they are lost the moment the terminal or tmux pane goes away, and
`nvidia-smi` only says something is running, not how it is doing.

Both do write TensorBoard events continuously, under the same scalar tag
(`Valid/f1_propeptides`), so that is the progress signal this reads. It works
from any shell, on runs started days ago, and on both the frozen and fine-tuned
paths.

Three states per run:

    DONE          test_metrics.json exists -- the run finished and was scored
    epoch N ...   in flight; shows the latest and best validation F1, plus how
                  long ago model.pt was last written, which is how long ago the
                  run last improved. A gap approaching --patience means it is
                  about to early-stop.
    starting up   directory exists but no events yet
'''
import argparse
import glob
import json
import os
import time

# Scalars carry a wall_time, so a finished run's duration and an in-flight run's
# seconds-per-epoch are both recoverable from the event file. That makes "how
# much longer" a measurement on this machine rather than a guess.

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAG = 'Valid/f1_propeptides'


def fmt(seconds):
    if seconds is None:
        return '?'
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f'{h}h{m:02d}m' if h else f'{m}m{s:02d}s'


# Hyperparameters that make two runs the same experiment. out_dir and seed are
# excluded deliberately -- differing on those is what makes them REPLICATES
# rather than different configurations.
GROUP_KEYS = ('embeddings_dir', 'embedding_dim', 'model', 'lr', 'batch_size',
              'dropout', 'conv_dropout', 'kernel_size', 'num_filters',
              'hidden_size', 'weight_decay', 'use_focal', 'epochs', 'patience',
              'lora_blocks', 'lora_r', 'lora_lr', 'max_len')


def config_key(run_dir):
    '''A hashable identity for the configuration, or None if unknowable.'''
    cfg = os.path.join(run_dir, 'config.json')
    if not os.path.isfile(cfg):
        return None
    c = json.load(open(cfg))
    return tuple((k, c[k]) for k in GROUP_KEYS if k in c)


def epochs_cap(run_dir):
    '''max epochs and patience from the run's own config.json, so the ETA uses
    the settings the run was actually launched with.'''
    cfg = os.path.join(run_dir, 'config.json')
    if not os.path.isfile(cfg):
        return None, None
    c = json.load(open(cfg))
    return c.get('epochs'), c.get('patience')


def read_scalars(run_dir):
    # A run restarted in place leaves several event files; the newest holds the
    # current attempt. Sorting by name works because the filename carries the
    # wall-clock start time.
    events = sorted(glob.glob(os.path.join(run_dir, 'events.out.tfevents.*')))
    if not events:
        return []
    acc = EventAccumulator(events[-1])
    acc.Reload()
    try:
        return acc.Scalars(TAG)
    except KeyError:          # writer opened, first validation not reached yet
        return []


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('runs', nargs='?', default='results/*',
                   help='Glob for run directories. Quote it so the shell does not '
                        'expand it.')
    args = p.parse_args()

    run_dirs = sorted(d for d in glob.glob(args.runs) if os.path.isdir(d))
    if not run_dirs:
        raise SystemExit(f'{args.runs} matched no directories.')

    done = {}
    for d in run_dirs:
        name = os.path.basename(d)
        test_json = os.path.join(d, 'test_metrics.json')
        if os.path.isfile(test_json):
            m = json.load(open(test_json))
            f1 = m['f1 propeptides']
            done.setdefault(config_key(d), []).append((name, f1))
            sc = read_scalars(d)
            # No events means the run was trained elsewhere and only its metrics
            # were copied here; say nothing rather than print '0 ep in ?'.
            took = (f'   [{len(sc)} ep in {fmt(sc[-1].wall_time - sc[0].wall_time)}]'
                    if len(sc) > 1 else '')
            print(f'{name:24} DONE   test F1 {f1:.4f}   '
                  f'P {m["precision propeptides"]:.4f}   R {m["recall propeptides"]:.4f}'
                  f'{took}')
            continue

        scalars = read_scalars(d)
        if not scalars:
            print(f'{name:24} starting up (no validation epoch yet)')
            continue

        best = max(s.value for s in scalars)
        ckpt = os.path.join(d, 'model.pt')
        # model.pt is written ONLY when validation improves, so its age is the
        # time since the last improvement -- i.e. the early-stopping counter,
        # recovered without the log.
        stall = ((time.time() - os.path.getmtime(ckpt)) / 60
                 if os.path.isfile(ckpt) else None)
        stall_s = f'{stall:.0f} min ago' if stall is not None else 'never'
        eta = ''
        if len(scalars) > 1:
            per = (scalars[-1].wall_time - scalars[0].wall_time) / (len(scalars) - 1)
            cap, pat = epochs_cap(d)
            left = []
            if cap:
                left.append(cap - len(scalars))
            # Early stopping usually fires first. model.pt's age divided by the
            # per-epoch time is how many epochs have passed without improving,
            # so patience minus that is the other candidate for "epochs left".
            if pat and stall is not None and per > 0:
                left.append(max(0, pat - round(stall * 60 / per)))
            if left:
                eta = f'   ~{fmt(min(left) * per)} left ({fmt(per)}/epoch)'
        print(f'{name:24} epoch {scalars[-1].step + 1:3d}   val_f1 {scalars[-1].value:.4f}   '
              f'best {best:.4f}   last improved {stall_s}{eta}')

    # Aggregate ONLY within a configuration. Averaging across configurations --
    # ESM-2 next to ESM3 next to ProstT5 -- produces a number that means nothing
    # and invites being quoted as though it did.
    import statistics
    groups = [(k, v) for k, v in done.items() if k is not None and len(v) > 1]
    if groups:
        print()
        # Label each group by the keys that actually DIFFER between groups. A
        # fixed template printed 'lr=0.0055 finetuned' for two different LoRA
        # groups and 'lr=0.0055 esm2' for two different ESM-2 groups -- identical
        # labels on distinct experiments, which is how numbers end up attributed
        # to the wrong recipe in a write-up.
        # embeddings_dir is skipped: its basename is already the first token of
        # every label, so repeating it in full only adds width.
        varying = [k for k in GROUP_KEYS
                   if k != 'embeddings_dir'
                   and len({dict(key).get(k) for key, _ in groups}) > 1]
        for key, runs in sorted(groups, key=lambda kv: -len(kv[1])):
            f1s = [f for _, f in runs]
            cfg = dict(key)
            emb = os.path.basename(str(cfg.get('embeddings_dir', '')).rstrip('/'))
            parts = [emb or 'finetuned']
            parts += ['{}={}'.format(k, cfg[k]) for k in varying if k in cfg]
            label = ' '.join(parts)
            members = ', '.join(n for n, _ in sorted(runs))
            print(f'{len(runs)} replicates of [{label}]: mean {statistics.mean(f1s):.4f}, '
                  f'sd {statistics.stdev(f1s):.4f}, min {min(f1s):.4f}, max {max(f1s):.4f}')
            print(f'    {members}')
    singles = sum(len(v) for k, v in done.items() if k is None or len(v) == 1)
    if singles:
        print(f'\n{singles} further finished run(s) are alone in their configuration '
              f'(or have no config.json) -- not aggregated.')


if __name__ == '__main__':
    main()
