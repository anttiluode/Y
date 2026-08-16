"""Gate 2 fairness audit before promoting any structured receiver.

The first 5-epoch smoke run showed grouped2 > dense on one Digits split. Before
spending a full multi-seed run on that bump, audit two implementation details:

1. low-rank reduction now uses variance-matched factor initialization;
2. Wu et al.'s fixed dendritic endpoint sums branches, whereas historical Y
   gates used their mean. Mean/sqrt-sum/sum have identical connectivity and
   parameters, so differences diagnose optimization scale rather than capacity.

This file is intentionally small and reuses Gate-2's paired training protocol.
"""

from __future__ import annotations

import argparse
import statistics

import torch
from scipy.stats import ttest_rel

from gate2_cost_locality import digits_split, seed_all, train_model
from y.efficient import make_control_mlp


def make_model(name: str, *, receiver_width: int, depth: int, budget_factor: int):
    common = dict(
        input_dim=64,
        receiver_width=receiver_width,
        depth=depth,
        budget_factor=budget_factor,
        classes=10,
    )
    if name == "dense":
        return make_control_mlp("dense", **common)
    if name.startswith("grouped"):
        return make_control_mlp("grouped", groups=int(name[7:]), **common)
    if name.startswith("lowrank"):
        return make_control_mlp("lowrank", rank=int(name[7:]), **common)
    if name.startswith("fixed_"):
        reduction = name.removeprefix("fixed_")
        return make_control_mlp("fixed", fixed_reduction=reduction, **common)
    raise ValueError(name)


def run(args):
    device = torch.device(args.device)
    names = [
        "dense",
        "grouped2",
        "grouped4",
        "lowrank4",
        "lowrank8",
        "lowrank16",
        "fixed_mean",
        "fixed_sqrt_sum",
        "fixed_sum",
    ]
    rows: dict[str, list[float]] = {name: [] for name in names}

    for split_seed in args.seeds:
        xtr, ytr, xte, yte = digits_split(split_seed)
        for name in names:
            seed_all(split_seed)
            model = make_model(
                name,
                receiver_width=args.receiver_width,
                depth=args.depth,
                budget_factor=args.budget_factor,
            )
            acc, sec = train_model(
                model,
                xtr,
                ytr,
                xte,
                yte,
                split_seed=split_seed,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
            )
            rows[name].append(acc)
            print(f"seed={split_seed:2d} {name:<15} acc={acc:.6f} sec={sec:.2f}")

    print("\nmean accuracy")
    for name, values in rows.items():
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        print(f"{name:<15} {statistics.fmean(values):.6f} +/- {sd:.6f}")

    if len(args.seeds) > 1:
        dense = rows["dense"]
        print("\npaired difference versus dense")
        for name, values in rows.items():
            if name == "dense":
                continue
            delta = statistics.fmean([a - b for a, b in zip(values, dense)])
            _t, p = ttest_rel(values, dense)
            print(f"{name:<15} {delta:+.6f}   p={p:.6g}")

        print("\nfixed scale-only pairwise differences")
        mean = rows["fixed_mean"]
        for name in ("fixed_sqrt_sum", "fixed_sum"):
            values = rows[name]
            delta = statistics.fmean([a - b for a, b in zip(values, mean)])
            _t, p = ttest_rel(values, mean)
            print(f"{name:<15} vs mean {delta:+.6f}   p={p:.6g}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--receiver-width", type=int, default=32)
    p.add_argument("--budget-factor", type=int, default=16)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.seeds = args.seeds[:1]
        args.epochs = min(args.epochs, 5)
    return args


if __name__ == "__main__":
    run(parse_args())
