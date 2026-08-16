"""Gate 2: cost/locality controls before any receiver routing.

Gate 1 found no interior sparse-receiver winner on Digits.  Gate 2 therefore
asks a more ordinary question:

    If rich learned pre-collapse reduction is required, can standard structured
    reducers preserve the dense bottleneck's accuracy while changing measured
    hardware cost?

Two experiments live here:

1. Accuracy gate on scikit-learn Digits with explicitly paired minibatch order.
2. CUDA microbenchmark for forward / forward+backward latency and peak allocated
   memory on larger receiver sizes where launch overhead does not completely
   dominate.

The CUDA measurements are wall-clock/allocation measurements only.  PyTorch's
profiler can expose kernels and allocations, but this script does NOT call those
numbers physical DRAM traffic.  Nsight/CUPTI-class counters are required for a
physical memory-traffic claim.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.stats import ttest_rel
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from y.efficient import (
    DenseBudgetBlock,
    FixedBranchBudgetBlock,
    GroupedReducerBlock,
    LowRankReducerBlock,
    block_budget_slack,
    block_parameter_count,
    hidden_budget,
    make_control_mlp,
)


@dataclass
class AccuracyResult:
    split_seed: int
    model: str
    receiver_width: int
    local_width: int
    reducer_weights: int
    block_params: int
    budget_slack: int
    test_accuracy: float
    train_seconds: float


@dataclass
class HardwareResult:
    model: str
    batch: int
    receiver_width: int
    depth: int
    dtype: str
    block_params: int
    local_width: int
    reducer_weights: int
    forward_ms: float
    forward_backward_ms: float | None
    peak_forward_bytes_delta: int | None
    peak_backward_bytes_delta: int | None
    logical_receiver_bytes: int
    logical_local_bytes_if_materialized: int


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def paired_loader(
    x: Tensor,
    y: Tensor,
    *,
    split_seed: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    generator = None
    if shuffle:
        generator = torch.Generator().manual_seed(split_seed + 424_242)
    return DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


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
    train_loader = paired_loader(
        x_train,
        y_train,
        split_seed=split_seed,
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = paired_loader(
        x_test,
        y_test,
        split_seed=split_seed,
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


def variant_specs(args: argparse.Namespace) -> list[tuple[str, str, int | None]]:
    specs: list[tuple[str, str, int | None]] = [("dense", "dense", None)]
    specs.extend((f"lowrank{rank}", "lowrank", rank) for rank in args.ranks)
    specs.extend((f"grouped{groups}", "grouped", groups) for groups in args.groups)
    specs.append(("fixed", "fixed", None))
    return specs


def make_mlp_from_spec(
    spec: tuple[str, str, int | None],
    *,
    input_dim: int,
    receiver_width: int,
    depth: int,
    budget_factor: int,
    classes: int,
) -> nn.Module:
    _name, kind, value = spec
    kwargs = dict(
        input_dim=input_dim,
        receiver_width=receiver_width,
        depth=depth,
        budget_factor=budget_factor,
        classes=classes,
    )
    if kind == "lowrank":
        return make_control_mlp(kind, rank=value, **kwargs)
    if kind == "grouped":
        return make_control_mlp(kind, groups=value, **kwargs)
    return make_control_mlp(kind, **kwargs)


def run_accuracy(args: argparse.Namespace) -> list[AccuracyResult]:
    device = torch.device(args.device)
    rows: list[AccuracyResult] = []
    specs = variant_specs(args)
    for split_seed in args.seeds:
        xtr, ytr, xte, yte = digits_split(split_seed)
        for spec in specs:
            name = spec[0]
            seed_all(split_seed)
            model = make_mlp_from_spec(
                spec,
                input_dim=64,
                receiver_width=args.receiver_width,
                depth=args.depth,
                budget_factor=args.budget_factor,
                classes=10,
            )
            block = model.blocks[0]
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
            rows.append(
                AccuracyResult(
                    split_seed=split_seed,
                    model=name,
                    receiver_width=args.receiver_width,
                    local_width=int(block.local_width),
                    reducer_weights=int(block.reducer_weights),
                    block_params=block_parameter_count(block),
                    budget_slack=block_budget_slack(block),
                    test_accuracy=acc,
                    train_seconds=sec,
                )
            )
    return rows


def print_accuracy(rows: list[AccuracyResult]) -> None:
    print(
        f"{'seed':>4} {'model':<12} {'local':>6} {'red_w':>8} {'params':>8} "
        f"{'slack':>6} {'acc':>8} {'sec':>8}"
    )
    for r in rows:
        print(
            f"{r.split_seed:>4} {r.model:<12} {r.local_width:>6} "
            f"{r.reducer_weights:>8} {r.block_params:>8} {r.budget_slack:>6} "
            f"{r.test_accuracy:>8.4f} {r.train_seconds:>8.2f}"
        )

    by_model = {name: [r.test_accuracy for r in rows if r.model == name]
                for name in sorted({r.model for r in rows})}
    print("\nmean accuracy")
    for name, values in by_model.items():
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        print(f"{name:<12} {statistics.fmean(values):.6f} +/- {sd:.6f}")

    if "dense" in by_model and len(by_model["dense"]) > 1:
        dense = by_model["dense"]
        print("\npaired difference versus dense")
        for name, values in by_model.items():
            if name == "dense" or len(values) != len(dense):
                continue
            diffs = [a - b for a, b in zip(values, dense)]
            _t, p = ttest_rel(values, dense)
            print(f"{name:<12} {statistics.fmean(diffs):+.6f}   p={p:.6g}")


def make_block(name: str, receiver_width: int, budget_factor: int) -> nn.Module:
    if name == "dense":
        return DenseBudgetBlock(receiver_width, budget_factor)
    if name == "fixed":
        return FixedBranchBudgetBlock(receiver_width, budget_factor)
    if name.startswith("lowrank"):
        return LowRankReducerBlock(receiver_width, budget_factor, int(name[7:]))
    if name.startswith("grouped"):
        return GroupedReducerBlock(receiver_width, budget_factor, int(name[7:]))
    raise ValueError(f"unknown model: {name}")


def dtype_from_name(name: str) -> torch.dtype:
    table = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return table[name]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {name}") from exc


def cuda_elapsed_ms(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / iters


def peak_cuda_delta(fn) -> int:
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated() - baseline)


def maybe_compile(module: nn.Module, args: argparse.Namespace) -> nn.Module:
    if not args.compile:
        return module
    try:
        return torch.compile(module, mode=args.compile_mode)
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"compile failed, continuing eager: {type(exc).__name__}: {exc}")
        return module


def run_hardware(args: argparse.Namespace) -> list[HardwareResult]:
    if not torch.cuda.is_available():
        raise RuntimeError("Gate-2 hardware benchmark requires CUDA")
    device = torch.device("cuda")
    dtype = dtype_from_name(args.bench_dtype)
    rows: list[HardwareResult] = []

    model_names = ["dense"]
    model_names += [f"lowrank{x}" for x in args.ranks]
    model_names += [f"grouped{x}" for x in args.groups]
    model_names += ["fixed"]

    for name in model_names:
        first_block = make_block(name, args.bench_receiver_width, args.budget_factor)
        block_params = block_parameter_count(first_block)
        local_width = int(first_block.local_width)
        reducer_weights = int(first_block.reducer_weights)

        for batch in args.bench_batches:
            seed_all(args.seed)
            blocks = [
                make_block(name, args.bench_receiver_width, args.budget_factor)
                for _ in range(args.bench_depth)
            ]
            core = nn.Sequential(*blocks).to(device=device, dtype=dtype).eval()
            core = maybe_compile(core, args)
            x = torch.randn(
                batch,
                args.bench_receiver_width,
                device=device,
                dtype=dtype,
                requires_grad=False,
            )
            elem_bytes = x.element_size()

            def fwd() -> Tensor:
                with torch.inference_mode():
                    return core(x)

            fwd_ms = cuda_elapsed_ms(
                fwd,
                warmup=args.bench_warmup,
                iters=args.bench_iters,
            )
            peak_fwd = peak_cuda_delta(fwd)

            fb_ms: float | None = None
            peak_bwd: int | None = None
            if args.bench_backward:
                core.train()
                x_train = x.detach().requires_grad_(True)

                def fwd_bwd() -> None:
                    core.zero_grad(set_to_none=True)
                    x_train.grad = None
                    y = core(x_train)
                    y.square().mean().backward()

                fb_ms = cuda_elapsed_ms(
                    fwd_bwd,
                    warmup=max(2, args.bench_warmup // 2),
                    iters=max(5, args.bench_iters // 4),
                )
                peak_bwd = peak_cuda_delta(fwd_bwd)
                core.eval()

            rows.append(
                HardwareResult(
                    model=name,
                    batch=batch,
                    receiver_width=args.bench_receiver_width,
                    depth=args.bench_depth,
                    dtype=args.bench_dtype,
                    block_params=block_params,
                    local_width=local_width,
                    reducer_weights=reducer_weights,
                    forward_ms=fwd_ms,
                    forward_backward_ms=fb_ms,
                    peak_forward_bytes_delta=peak_fwd,
                    peak_backward_bytes_delta=peak_bwd,
                    logical_receiver_bytes=batch * args.bench_receiver_width * elem_bytes,
                    logical_local_bytes_if_materialized=batch * local_width * elem_bytes,
                )
            )
            del core, x
            torch.cuda.empty_cache()
    return rows


def print_hardware(rows: list[HardwareResult]) -> None:
    print(
        f"{'model':<12} {'batch':>6} {'local':>6} {'red_w':>8} "
        f"{'fwd_ms':>10} {'fb_ms':>10} {'peak_fwd_MB':>12} {'peak_fb_MB':>11}"
    )
    for r in rows:
        fb = "-" if r.forward_backward_ms is None else f"{r.forward_backward_ms:.4f}"
        pf = "-" if r.peak_forward_bytes_delta is None else f"{r.peak_forward_bytes_delta/2**20:.3f}"
        pb = "-" if r.peak_backward_bytes_delta is None else f"{r.peak_backward_bytes_delta/2**20:.3f}"
        print(
            f"{r.model:<12} {r.batch:>6} {r.local_width:>6} {r.reducer_weights:>8} "
            f"{r.forward_ms:>10.4f} {fb:>10} {pf:>12} {pb:>11}"
        )


def profile_one(args: argparse.Namespace) -> None:
    if not args.profile_model:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("profiling requires CUDA")
    from torch.profiler import ProfilerActivity, profile

    dtype = dtype_from_name(args.bench_dtype)
    model = nn.Sequential(
        *[
            make_block(args.profile_model, args.bench_receiver_width, args.budget_factor)
            for _ in range(args.bench_depth)
        ]
    ).cuda().to(dtype=dtype).eval()
    x = torch.randn(
        args.profile_batch,
        args.bench_receiver_width,
        device="cuda",
        dtype=dtype,
    )
    for _ in range(5):
        with torch.inference_mode():
            model(x)
    torch.cuda.synchronize()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        for _ in range(10):
            with torch.inference_mode():
                model(x)
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25))
    if args.trace:
        path = Path(args.trace)
        path.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(path))
        print(f"wrote profiler trace: {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--receiver-width", type=int, default=32)
    p.add_argument("--budget-factor", type=int, default=16)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--ranks", type=int, nargs="+", default=[4, 8, 16])
    p.add_argument("--groups", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--hardware", action="store_true")
    p.add_argument("--bench-receiver-width", type=int, default=256)
    p.add_argument("--bench-depth", type=int, default=8)
    p.add_argument("--bench-batches", type=int, nargs="+", default=[1, 8, 32, 128, 512])
    p.add_argument("--bench-warmup", type=int, default=20)
    p.add_argument("--bench-iters", type=int, default=100)
    p.add_argument("--bench-dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    p.add_argument("--bench-backward", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--compile-mode", default="default")
    p.add_argument("--profile-model", default="")
    p.add_argument("--profile-batch", type=int, default=128)
    p.add_argument("--trace", default="")

    p.add_argument("--accuracy-only", action="store_true")
    p.add_argument("--hardware-only", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.seeds = args.seeds[:1]
        args.epochs = min(args.epochs, 5)
        args.ranks = args.ranks[:1]
        args.groups = args.groups[:2]
    return args


def main() -> None:
    args = parse_args()
    output: dict[str, object] = {
        "nominal_hidden_budget": hidden_budget(args.receiver_width, args.budget_factor),
    }

    run_acc = not args.hardware_only
    run_hw = args.hardware and not args.accuracy_only or args.hardware_only

    if run_acc:
        acc_rows = run_accuracy(args)
        output["accuracy"] = [asdict(r) for r in acc_rows]
        if not args.json:
            print_accuracy(acc_rows)

    if run_hw:
        hw_rows = run_hardware(args)
        output["hardware"] = [asdict(r) for r in hw_rows]
        output["gpu"] = torch.cuda.get_device_name(0)
        if not args.json:
            print("\nCUDA hardware measurements")
            print(f"device: {torch.cuda.get_device_name(0)}")
            print_hardware(hw_rows)
        profile_one(args)

    if args.json:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
