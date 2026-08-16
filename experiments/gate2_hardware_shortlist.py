"""Gate 2 hardware shortlist after the receiver-scale correction.

This is the hardware-facing instrument Y should run before inventing another
receiver mechanism.

The capacity audit found no compelling accuracy distinction among dense,
grouped, and fixed local aggregation once positive receiver scale was
controlled.  Hardware is therefore measured on a deliberately short list:

    dense, fixed_sum, grouped2, grouped4, lowrank16

Two views are reported:

``micro``
    One hidden block repeatedly receives the same input.  This permits the
    exact paper-form raw branch SUM without numerically compounding that scale
    through a deep stack.

``stack``
    Several independent hidden blocks are chained, with parameter-free
    LayerNorm after every block exactly as in the Gate-2 scale-control audit.
    All candidates pay the same normalization overhead.

The script measures CUDA-event wall clock and PyTorch active-allocation peaks.
It does NOT infer physical DRAM bytes or energy.  Those require hardware
counters / an appropriate profiler.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import torch
from torch import Tensor, nn

from y.efficient import (
    DenseBudgetBlock,
    FixedBranchBudgetBlock,
    GroupedReducerBlock,
    LowRankReducerBlock,
    block_parameter_count,
)


@dataclass
class BenchRow:
    mode: str
    candidate: str
    batch: int
    receiver_width: int
    budget_factor: int
    depth: int
    dtype: str
    compiled: bool
    block_params: int
    total_params: int
    local_width: int
    reducer_weights: int
    learned_macs_per_sample_per_block: int
    logical_receiver_bytes_per_boundary: int
    logical_local_bytes_if_materialized_per_block: int
    forward_ms: float
    forward_backward_ms: float | None
    peak_forward_active_bytes_delta: int
    peak_forward_backward_active_bytes_delta: int | None


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dtype_from_name(name: str) -> torch.dtype:
    table = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return table[name]
    except KeyError as exc:
        raise ValueError(name) from exc


def make_block(name: str, receiver_width: int, budget_factor: int) -> nn.Module:
    if name == "dense":
        return DenseBudgetBlock(receiver_width, budget_factor)
    if name == "fixed_sum":
        return FixedBranchBudgetBlock(
            receiver_width,
            budget_factor,
            reduction="sum",
        )
    if name == "fixed_sqrt_sum":
        return FixedBranchBudgetBlock(
            receiver_width,
            budget_factor,
            reduction="sqrt_sum",
        )
    if name == "fixed_mean":
        return FixedBranchBudgetBlock(
            receiver_width,
            budget_factor,
            reduction="mean",
        )
    if name.startswith("grouped"):
        return GroupedReducerBlock(
            receiver_width,
            budget_factor,
            groups=int(name.removeprefix("grouped")),
        )
    if name.startswith("lowrank"):
        return LowRankReducerBlock(
            receiver_width,
            budget_factor,
            rank=int(name.removeprefix("lowrank")),
        )
    raise ValueError(f"unknown candidate: {name}")


def make_core(
    *,
    mode: str,
    candidate: str,
    receiver_width: int,
    budget_factor: int,
    depth: int,
) -> tuple[nn.Module, nn.Module, int]:
    """Return (core, representative_block, effective_depth)."""
    first = make_block(candidate, receiver_width, budget_factor)
    if mode == "micro":
        return first, first, 1
    if mode != "stack":
        raise ValueError(f"unknown mode: {mode}")

    modules: list[nn.Module] = []
    representative = first
    for i in range(depth):
        block = first if i == 0 else make_block(
            candidate, receiver_width, budget_factor
        )
        modules.append(block)
        modules.append(
            nn.LayerNorm(receiver_width, elementwise_affine=False)
        )
    return nn.Sequential(*modules), representative, depth


def maybe_compile(module: nn.Module, args: argparse.Namespace) -> nn.Module:
    if not args.compile:
        return module
    return torch.compile(module, mode=args.compile_mode)


def cuda_time_ms(fn: Callable[[], object], warmup: int, iters: int) -> float:
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


def active_peak_delta(fn: Callable[[], object]) -> int:
    torch.cuda.synchronize()
    baseline = int(torch.cuda.memory_allocated())
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    peak = int(torch.cuda.max_memory_allocated())
    return max(0, peak - baseline)


def bench_one(
    *,
    mode: str,
    candidate: str,
    batch: int,
    args: argparse.Namespace,
) -> BenchRow:
    dtype = dtype_from_name(args.dtype)
    seed_all(args.seed)
    core, block, effective_depth = make_core(
        mode=mode,
        candidate=candidate,
        receiver_width=args.receiver_width,
        budget_factor=args.budget_factor,
        depth=args.depth,
    )
    core = core.cuda().to(dtype=dtype)
    core.train(args.backward)
    core = maybe_compile(core, args)

    x = torch.randn(
        batch,
        args.receiver_width,
        device="cuda",
        dtype=dtype,
    ) * args.input_scale
    element_bytes = x.element_size()

    def forward() -> Tensor:
        with torch.inference_mode():
            return core(x)

    # Compilation and lazy CUDA initialization must be outside measurements.
    for _ in range(args.warmup):
        forward()
    torch.cuda.synchronize()

    fwd_ms = cuda_time_ms(forward, args.warmup, args.iters)
    peak_fwd = active_peak_delta(forward)

    fb_ms: float | None = None
    peak_fb: int | None = None
    if args.backward:
        core.train()
        x_train = x.detach().requires_grad_(True)

        def forward_backward() -> None:
            core.zero_grad(set_to_none=True)
            x_train.grad = None
            y = core(x_train)
            # Mean keeps the scalar loss scale independent of batch / R.
            y.float().square().mean().backward()

        # Warm gradients / compiled backward before timing and peak reset.
        for _ in range(max(2, args.warmup // 2)):
            forward_backward()
        torch.cuda.synchronize()
        fb_ms = cuda_time_ms(
            forward_backward,
            max(2, args.warmup // 2),
            max(5, args.iters // 4),
        )
        peak_fb = active_peak_delta(forward_backward)

    block_params = block_parameter_count(block)
    total_params = sum(p.numel() for p in core.parameters())
    row = BenchRow(
        mode=mode,
        candidate=candidate,
        batch=batch,
        receiver_width=args.receiver_width,
        budget_factor=args.budget_factor,
        depth=effective_depth,
        dtype=args.dtype,
        compiled=args.compile,
        block_params=block_params,
        total_params=total_params,
        local_width=int(block.local_width),
        reducer_weights=int(block.reducer_weights),
        learned_macs_per_sample_per_block=block_params,
        logical_receiver_bytes_per_boundary=(
            batch * args.receiver_width * element_bytes
        ),
        logical_local_bytes_if_materialized_per_block=(
            batch * int(block.local_width) * element_bytes
        ),
        forward_ms=fwd_ms,
        forward_backward_ms=fb_ms,
        peak_forward_active_bytes_delta=peak_fwd,
        peak_forward_backward_active_bytes_delta=peak_fb,
    )

    del core, x
    torch.cuda.empty_cache()
    return row


def print_rows(rows: list[BenchRow]) -> None:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"{'mode':<6} {'candidate':<15} {'B':>5} {'H':>6} {'redW':>9} "
        f"{'fwd_ms':>9} {'fb_ms':>9} {'peakF_MB':>10} {'peakFB_MB':>10}"
    )
    for r in rows:
        fb = "-" if r.forward_backward_ms is None else f"{r.forward_backward_ms:.4f}"
        peak_fb = (
            "-"
            if r.peak_forward_backward_active_bytes_delta is None
            else f"{r.peak_forward_backward_active_bytes_delta / 2**20:.3f}"
        )
        print(
            f"{r.mode:<6} {r.candidate:<15} {r.batch:>5} {r.local_width:>6} "
            f"{r.reducer_weights:>9} {r.forward_ms:>9.4f} {fb:>9} "
            f"{r.peak_forward_active_bytes_delta / 2**20:>10.3f} {peak_fb:>10}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--candidates",
        nargs="+",
        default=["dense", "fixed_sum", "grouped2", "grouped4", "lowrank16"],
    )
    p.add_argument("--modes", nargs="+", choices=["micro", "stack"], default=["micro", "stack"])
    p.add_argument("--receiver-width", type=int, default=256)
    p.add_argument("--budget-factor", type=int, default=16)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32, 128, 512])
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    p.add_argument("--input-scale", type=float, default=0.25)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--backward", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--compile-mode", default="default")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.candidates = args.candidates[:3]
        args.modes = ["micro"]
        args.batches = [32]
        args.warmup = min(args.warmup, 3)
        args.iters = min(args.iters, 10)
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("gate2_hardware_shortlist.py requires CUDA")

    rows: list[BenchRow] = []
    for mode in args.modes:
        for candidate in args.candidates:
            for batch in args.batches:
                rows.append(
                    bench_one(
                        mode=mode,
                        candidate=candidate,
                        batch=batch,
                        args=args,
                    )
                )

    if args.json:
        print(
            json.dumps(
                {
                    "gpu": torch.cuda.get_device_name(0),
                    "torch": torch.__version__,
                    "rows": [asdict(r) for r in rows],
                },
                indent=2,
            )
        )
    else:
        print_rows(rows)
        print("\nNOTE: allocation deltas are PyTorch active-memory peaks, not DRAM traffic.")


if __name__ == "__main__":
    main()
