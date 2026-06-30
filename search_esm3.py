#!/usr/bin/env python3
"""Optuna hyperparameter search for the full (peptide + propeptide) ESM3 model.

Single-fold protocol: train on partitions [0, 1, 2], select on val [3], test [4] held out.
Objective = val_f1 = mean(peptide F1, propeptide F1)  -- the paper's stopping metric.
Search space = paper Supplementary Table S1. Test is *recorded* but NEVER selected on.

Run (3 parallel workers sharing one SQLite study -> uses all cores):
    pip install optuna
    mkdir -p logs search_esm3
    for i in 0 1 2; do
      OMP_NUM_THREADS=14 MKL_NUM_THREADS=14 nice -n 5 nohup \
        python search_esm3.py --n-trials 12 --epochs 30 \
        --embeddings-dir /data/apostolos/embeddings/esm3 \
        > logs/search_$i.log 2>&1 &
    done

Inspect best-so-far at any time (safe while running):
    python search_esm3.py --report
"""
import argparse
import json
import os

import optuna

from src.train_loop_crf import train


def objective(trial, emb, epochs, out_root):
    params = dict(
        lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        batch_size=trial.suggest_int("batch_size", 10, 100, step=10),
        dropout=trial.suggest_float("dropout", 0.0, 0.7),
        conv_dropout=trial.suggest_float("conv_dropout", 0.0, 0.7),
        kernel_size=trial.suggest_categorical("kernel_size", [1, 3, 5]),
        num_filters=trial.suggest_int("num_filters", 40, 128, step=8),
        hidden_size=trial.suggest_int("hidden_size", 16, 192, step=16),
    )
    out = os.path.join(out_root, f"trial_{trial.number}")
    os.makedirs(out, exist_ok=True)
    args = argparse.Namespace(
        embeddings_dir=emb,
        data_file="data/labeled_sequences.csv",
        partitioning_file="data/graphpart_assignments.csv",
        embedding="precomputed",
        embedding_dim=1536,
        model="lstmcnncrf",
        out_dir=out,
        epochs=epochs,
        label_type="multistate_with_propeptides",
        **params,
    )
    best_val, test = train(args, [0, 1, 2], [3], [4])
    trial.set_user_attr("test_f1_pep", test.get("f1 peptides"))
    trial.set_user_attr("test_f1_pro", test.get("f1 propeptides"))
    return (best_val["f1 peptides"] + best_val["f1 propeptides"]) / 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings-dir", default="/data/apostolos/embeddings/esm3")
    p.add_argument("--n-trials", type=int, default=12)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--storage", default="sqlite:///esm3_hp.db")
    p.add_argument("--study", default="esm3_full_hp")
    p.add_argument("--out-root", default="search_esm3")
    p.add_argument("--report", action="store_true")
    a = p.parse_args()

    study = optuna.create_study(
        direction="maximize",
        study_name=a.study,
        storage=a.storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(multivariate=True),
    )

    if a.report:
        done = [t for t in study.trials if t.value is not None]
        if not done:
            print("no completed trials yet")
            return
        b = study.best_trial
        print(f"BEST trial {b.number}  val_f1={b.value:.4f}")
        print("params:", json.dumps(b.params, indent=2))
        print("its test:", b.user_attrs)
        print(f"\nTop 8 of {len(done)} completed trials by val_f1:")
        for t in sorted(done, key=lambda t: -t.value)[:8]:
            print(
                f"  #{t.number:2d} val={t.value:.4f}  lr={t.params['lr']:.4g} "
                f"bs={t.params['batch_size']} do={t.params['dropout']:.2f} "
                f"cdo={t.params['conv_dropout']:.2f} k={t.params['kernel_size']} "
                f"nf={t.params['num_filters']} hs={t.params['hidden_size']}"
            )
        return

    os.makedirs(a.out_root, exist_ok=True)
    study.optimize(
        lambda t: objective(t, a.embeddings_dir, a.epochs, a.out_root),
        n_trials=a.n_trials,
    )


if __name__ == "__main__":
    main()
