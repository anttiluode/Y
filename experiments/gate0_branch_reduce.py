"""Gate 0: can rich local compute buy a narrower communicated interface?

Compares three narrow mechanisms at matched receiver width and hidden weight
budget: fixed branch reduction, an ordinary learned bottleneck, and a tempting
post-collapse mixer negative control.

The communication metric is a logical activation-width proxy, not measured
DRAM traffic. Real hardware claims require a fused kernel + profiler gate.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from y import BottleneckMLP, BranchMLP, MixedBranchMLP, PointMLP, matched_receiver_width


@dataclass
class Result:
    model: str
    branches: int
    receiver_width: int
    params: int
    hidden_core_weights_per_block: int
    receiver_values_per_sample: int
    receiver_ratio_vs_point: float
    test_accuracy: float
    train_seconds: float


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_teacher_data(n_train: int, n_test: int, input_dim: int, classes: int, seed: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    g = torch.Generator().manual_seed(seed + 101)
    total = n_train + n_test
    x = torch.randn(total, input_dim, generator=g)
    teacher_width = max(64, input_dim * 4)
    w1 = torch.randn(input_dim, teacher_width, generator=g) / input_dim**0.5
    b1 = torch.randn(teacher_width, generator=g) * 0.15
    w2 = torch.randn(teacher_width, classes, generator=g) / teacher_width**0.5
    h = torch.relu(x @ w1 + b1)
    logits = h @ w2 + 0.25 * torch.sin(h[:, :classes] * 1.7)
    y = logits.argmax(dim=-1)
    return x[:n_train], y[:n_train], x[n_train:], y[n_train:]


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    right = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=-1)
            right += int((pred == y).sum())
            total += y.numel()
    return right / max(1, total)


def train_model(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, device: torch.device, epochs: int, lr: float) -> tuple[float, float]:
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
    return accuracy(model, test_loader, device), time.perf_counter() - start


def append_result(results, model_name, k, width, model, acc, sec, point_receiver_values, depth):
    receiver_values = width * depth
    results.append(Result(
        model=model_name,
        branches=k,
        receiver_width=width,
        params=count_params(model),
        hidden_core_weights_per_block=k * width * width,
        receiver_values_per_sample=receiver_values,
        receiver_ratio_vs_point=receiver_values / point_receiver_values,
        test_accuracy=acc,
        train_seconds=sec,
    ))


def run(args: argparse.Namespace) -> list[Result]:
    seed_all(args.seed)
    device = torch.device(args.device)
    xtr, ytr, xte, yte = make_teacher_data(args.train_samples, args.test_samples, args.input_dim, args.classes, args.seed)
    train_loader = DataLoader(TensorDataset(xtr, ytr), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(xte, yte), batch_size=args.batch_size, shuffle=False)

    results: list[Result] = []
    point = PointMLP(args.input_dim, args.point_width, args.depth, args.classes)
    acc, sec = train_model(point, train_loader, test_loader, device, args.epochs, args.lr)
    point_receiver_values = args.point_width * args.depth
    results.append(Result(
        model="point", branches=1, receiver_width=args.point_width,
        params=count_params(point), hidden_core_weights_per_block=args.point_width**2,
        receiver_values_per_sample=point_receiver_values, receiver_ratio_vs_point=1.0,
        test_accuracy=acc, train_seconds=sec,
    ))

    variants = (
        ("branch", BranchMLP),
        ("bneck", BottleneckMLP),
        ("postmix", MixedBranchMLP),
    )
    for k in args.branches:
        width = matched_receiver_width(args.point_width, k)
        for name, cls in variants:
            seed_all(args.seed)
            model = cls(args.input_dim, width, args.depth, k, args.classes)
            acc, sec = train_model(model, train_loader, test_loader, device, args.epochs, args.lr)
            append_result(results, f"{name}_k{k}", k, width, model, acc, sec, point_receiver_values, args.depth)
    return results


def print_table(results: list[Result]) -> None:
    print(f"{'model':<12} {'K':>3} {'recv':>6} {'params':>10} {'core_w':>10} {'traffic':>8} {'acc':>8} {'train_s':>9}")
    for r in results:
        print(f"{r.model:<12} {r.branches:>3} {r.receiver_width:>6} {r.params:>10} {r.hidden_core_weights_per_block:>10} {r.receiver_ratio_vs_point:>8.3f} {r.test_accuracy:>8.4f} {r.train_seconds:>9.2f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dim", type=int, default=32)
    p.add_argument("--classes", type=int, default=8)
    p.add_argument("--point-width", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--branches", type=int, nargs="+", default=[4, 16])
    p.add_argument("--train-samples", type=int, default=8192)
    p.add_argument("--test-samples", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--json", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.train_samples = min(args.train_samples, 2048)
        args.test_samples = min(args.test_samples, 512)
        args.epochs = min(args.epochs, 4)
    return args


if __name__ == "__main__":
    args = parse_args()
    rows = run(args)
    print(json.dumps([asdict(r) for r in rows], indent=2) if args.json else "") if args.json else print_table(rows)
