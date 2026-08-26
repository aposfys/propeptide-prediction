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

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAG = 'Valid/f1_propeptides'


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

    done = []
    for d in run_dirs:
        name = os.path.basename(d)
        test_json = os.path.join(d, 'test_metrics.json')
        if os.path.isfile(test_json):
            m = json.load(open(test_json))
            f1 = m['f1 propeptides']
            done.append(f1)
            print(f'{name:24} DONE   test F1 {f1:.4f}   '
                  f'P {m["precision propeptides"]:.4f}   R {m["recall propeptides"]:.4f}')
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
        print(f'{name:24} epoch {scalars[-1].step + 1:3d}   val_f1 {scalars[-1].value:.4f}   '
              f'best {best:.4f}   last improved {stall_s}')

    if len(done) > 1:
        import statistics
        print(f'\n{len(done)} finished: mean test F1 {statistics.mean(done):.4f}'
              + (f', sd {statistics.stdev(done):.4f}' if len(done) > 1 else '')
              + f', min {min(done):.4f}, max {max(done):.4f}')


if __name__ == '__main__':
    main()
