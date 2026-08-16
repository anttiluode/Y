"""Gate 1: map the pre-collapse receiver frontier on a standard dataset.

The question is deliberately narrower than "do dendrites help?" At a fixed
communicated receiver width R and fixed hidden learned-weight budget K*R^2,
how much of that budget should create private nonlinear features and how much
should learn the receiver that compresses those features before communication?

The external dataset is scikit-learn Digits, which ships with scikit-learn and
requires no network download.

Important pairing rule: every model in a split receives its own DataLoader with
an explicit identical shuffle generator. Different model initializations may
consume different amounts of RNG, so relying on the global RNG would silently
give architectures different minibatch orders.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from y import (
    BottleneckMLP,
    BranchMLP,
    PointMLP,
    PrecollapseReceiverMLP,
    matched_receiver_width,
)


@dataclass
class Result:
    split_seed: int
    model: str
    receiver_width: int
    receiver_ratio_vs_point: float
    local_width: int
    reducer_fanin: int
    reducer_budget_fraction: float | None
    params: int
    test_accuracy: float
    train_seconds: float


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    right = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            right += int((model(x).argmax(dim=-1) == y).sum())
            total += y.numel()
    return right / max(1, total)


def train_model(
    model: nn.Module,
    x_train: Tensor,
    y_train: Tensor,
    x_test: Tensor,
    y_test: Tensor,
    *,
    split_seed: int,
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
) -> tuple[float, float]:
    # Crucial: minibatch order is paired across architectures and independent
    # of how much RNG a model consumed while initializing differently-shaped
    # parameter tensors.
    shuffle_generator = torch.Generator().manual_seed(split_seed + 424_242)
    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        generator=shuffle_generator,
    )
    test_loader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=max(batch_size, 256),
        shuffle=False,
    )

    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    start = time.perf_counter()
    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
    elapsed = time.perf_counter() - start
    return accuracy(model, test_loader, device), elapsed


def digits_split(seed: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    digits = load_digits()
    x_train, x_test, y_train, y_test = train_test_split(
        digits.data.astype(np.float32) / 16.0,
        digits.target.astype(np.int64),
        test_size=0.30,
        random_state=seed,
        stratify=digits.target,
    )
    return (
        torch.tensor(x_train),
        torch.tensor(y_train),
        torch.tensor(x_test),
        torch.tensor(y_test),
    )


def run(args: argparse.Namespace) -> list[Result]:
    device = torch.device(args.device)
    receiver_width = matched_receiver_width(args.point_width, args.branches)
    results: list[Result] = []

    for split_seed in args.seeds:
        xtr, ytr, xte, yte = digits_split(split_seed)

        specs: list[tuple[str, callable, int, int, float | None]] = [
            (
                "point",
                lambda: PointMLP(64, args.point_width, args.depth, 10),
                args.point_width,
                args.point_width,
                None,
            ),
            (
                "branch",
                lambda: BranchMLP(64, receiver_width, args.depth, args.branches, 10),
                receiver_width,
                args.branches * receiver_width,
                0.0,
            ),
            (
                "bneck",
                lambda: BottleneckMLP(64, receiver_width, args.depth, args.branches, 10),
                receiver_width,
                args.branches * receiver_width // 2,
                0.5,
            ),
        ]
        for fanin in args.fanins:
            specs.append(
                (
                    f"pre{fanin}",
                    lambda fanin=fanin: PrecollapseReceiverMLP(
                        input_dim=64,
                        receiver_width=receiver_width,
                        depth=args.depth,
                        branches=args.branches,
                        reducer_fanin=fanin,
                        classes=10,
                        index_seed=args.index_seed,
                    ),
                    receiver_width,
                    args.branches * receiver_width - fanin,
                    fanin / (args.branches * receiver_width),
                )
            )

        for name, make_model, width, local_width, reducer_fraction in specs:
            seed_all(split_seed)
            model = make_model()
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
            fanin = 0
            if name.startswith("pre"):
                fanin = int(name[3:])
            elif name == "bneck":
                fanin = local_width
            results.append(
                Result(
                    split_seed=split_seed,
                    model=name,
                    receiver_width=width,
                    receiver_ratio_vs_point=width / args.point_width,
                    local_width=local_width,
                    reducer_fanin=fanin,
                    reducer_budget_fraction=reducer_fraction,
                    params=count_params(model),
                    test_accuracy=acc,
                    train_seconds=sec,
                )
            )
    return results


def print_results(rows: list[Result]) -> None:
    print(
        f"{'seed':>4} {'model':<10} {'recv':>5} {'local':>6} {'fanin':>6} "
        f"{'red%':>7} {'params':>8} {'acc':>8} {'sec':>8}"
    )
    for r in rows:
        red = "-" if r.reducer_budget_fraction is None else f"{100*r.reducer_budget_fraction:.1f}"
        print(
            f"{r.split_seed:>4} {r.model:<10} {r.receiver_width:>5} {r.local_width:>6} "
            f"{r.reducer_fanin:>6} {red:>7} {r.params:>8} {r.test_accuracy:>8.4f} "
            f"{r.train_seconds:>8.2f}"
        )

    print("\nmean accuracy")
    for model in sorted({r.model for r in rows}):
        values = [r.test_accuracy for r in rows if r.model == model]
        mean = statistics.fmean(values)
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        print(f"{model:<10} {mean:.6f} +/- {sd:.6f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--point-width", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--branches", type=int, default=16)
    p.add_argument("--fanins", type=int, nargs="+", default=[32, 64, 128, 256])
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--index-seed", type=int, default=777)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--json", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.seeds = args.seeds[:1]
        args.epochs = min(args.epochs, 5)
    return args


if __name__ == "__main__":
    args = parse_args()
    rows = run(args)
    if args.json:
        print(json.dumps([asdict(r) for r in rows], indent=2))
    else:
        print_results(rows)
