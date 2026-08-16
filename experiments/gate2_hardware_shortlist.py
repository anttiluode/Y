"""Gate 2 hardware shortlist after the receiver-scale correction.

This instrument separates two questions that must not be conflated.

PAPER REPLICATION AXIS
----------------------
At reference narrow width R and budget factor K:

    wide_point:  D -> D, where D = R*sqrt(K)
    fixed_sum:   R -> K*R -> grouped nonlinear SUM -> R

Both use K*R^2 learned weights/MACs per sample.  The wide point layer transmits
sqrt(K) times more activations at the block boundary.  This is the closest
single-layer analogue here to Wu et al.'s equal-compute point-vs-dendritic
comparison.

Y PRACTICAL CONTROL AXIS
------------------------
The stronger engineering question is whether dendritic/local topology beats
ordinary ways to keep the *same narrow R-wide boundary*:

    dense, grouped2, grouped4, lowrank16, fixed_sum

All expose R values and spend the same nominal learned-weight budget.  If a
boring dense bottleneck is faster/cheaper, narrow logical communication alone
is not a Y result.

Two execution views are supported:

``micro``
    One hidden block repeatedly receives the same input.  Exact paper-form raw
    branch SUM is safe because its scale is not compounded through depth.

``stack``
    Several independent hidden blocks are chained, with parameter-free
    LayerNorm after every block exactly as in the Gate-2 scale-control audit.
    All candidates pay the same normalization operation relative to their own
    boundary width.

The most important sweep is *size*.  The motivating paper's GPU analysis says
small matrices can remain cache-resident and show little advantage; the
predicted global-memory benefit becomes visible once working sets exceed cache.
Therefore this script sweeps reference R rather than reporting one cute number.

Measurements here are CUDA-event wall clock and PyTorch active-allocation
peaks.  They are NOT physical DRAM bytes or energy.  Use Nsight Compute / CUPTI
or an equivalent hardware-counter profiler for those claims.
"""

from __future__ import annotations

import argparse
import json
import math
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


class WidePointBudgetBlock(nn.Module):
    """Paper-axis point-neuron control at equal learned complexity.

    If the dendritic/local layer has reference receiver R and K branches, the
    equal-complexity point layer has width D = R*sqrt(K), because

        D^2 = K*R^2.

    K must therefore be a perfect square for this exact integer control.
    """

    kind = "wide_point"

    def __init__(self, reference_receiver_width: int, budget_factor: int) -> None:
        super().__init__()
        root = int(math.isqrt(budget_factor))
        if root * root != budget_factor:
            raise ValueError("wide_point requires budget_factor to be a perfect square")
        self.reference_receiver_width = int(reference_receiver_width)
        self.budget_factor = int(budget_factor)
        self.boundary_width = self.reference_receiver_width * root
        self.local_width = self.boundary_width
        self.reducer_weights = 0
        self.linear = nn.Linear(self.boundary_width, self.boundary_width, bias=False)
        self.act = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.linear(x))


@dataclass
class BenchRow:
    mode: str
    candidate: str
    batch: int
    reference_receiver_width: int
    input_width: int
    boundary_width: int
    boundary_ratio_vs_wide_point: float
    budget_factor: int
    depth: int
    dtype: str
    compiled: bool
    block_params: int
    total_params: int
    local_width: int
    reducer_weights: int
    learned_macs_per_sample_per_block: int
    logical_boundary_bytes: int
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


def wide_point_width(receiver_width: int, budget_factor: int) -> int:
    root = int(math.isqrt(budget_factor))
    if root * root != budget_factor:
        raise ValueError("budget_factor must be a perfect square")
    return receiver_width * root


def make_block(name: str, receiver_width: int, budget_factor: int) -> nn.Module:
    if name == "wide_point":
        return WidePointBudgetBlock(receiver_width, budget_factor)
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


def boundary_width_for(
    candidate: str,
    receiver_width: int,
    budget_factor: int,
) -> int:
    if candidate == "wide_point":
        return wide_point_width(receiver_width, budget_factor)
    return receiver_width


def make_core(
    *,
    mode: str,
    candidate: str,
    receiver_width: int,
    budget_factor: int,
    depth: int,
) -> tuple[nn.Module, nn.Module, int, int]:
    """Return core, representative block, effective depth, actual boundary width."""
    first = make_block(candidate, receiver_width, budget_factor)
    boundary_width = boundary_width_for(candidate, receiver_width, budget_factor)
    if mode == "micro":
        return first, first, 1, boundary_width
    if mode != "stack":
        raise ValueError(f"unknown mode: {mode}")

    modules: list[nn.Module] = []
    representative = first
    for i in range(depth):
        block = first if i == 0 else make_block(
            candidate, receiver_width, budget_factor
        )
        modules.append(block)
        # Parameter-free scale control. Wide point pays LN over D; narrow
        # candidates pay LN over R, which is part of the boundary-size story.
        modules.append(
            nn.LayerNorm(boundary_width, elementwise_affine=False)
        )
    return nn.Sequential(*modules), representative, depth, boundary_width


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
    receiver_width: int,
    batch: int,
    args: argparse.Namespace,
) -> BenchRow:
    dtype = dtype_from_name(args.dtype)
    seed_all(args.seed)
    core, block, effective_depth, boundary_width = make_core(
        mode=mode,
        candidate=candidate,
        receiver_width=receiver_width,
        budget_factor=args.budget_factor,
        depth=args.depth,
    )
    total_params = sum(p.numel() for p in core.parameters())
    core = core.cuda().to(dtype=dtype)
    core.train(args.backward)
    core = maybe_compile(core, args)

    x = torch.randn(
        batch,
        boundary_width,
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
            # Mean keeps scalar loss scale independent of batch / width.
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
    local_width = int(getattr(block, "local_width"))
    reducer_weights = int(getattr(block, "reducer_weights", 0))
    paper_wide_width = wide_point_width(receiver_width, args.budget_factor)

    row = BenchRow(
        mode=mode,
        candidate=candidate,
        batch=batch,
        reference_receiver_width=receiver_width,
        input_width=boundary_width,
        boundary_width=boundary_width,
        boundary_ratio_vs_wide_point=boundary_width / paper_wide_width,
        budget_factor=args.budget_factor,
        depth=effective_depth,
        dtype=args.dtype,
        compiled=args.compile,
        block_params=block_params,
        total_params=total_params,
        local_width=local_width,
        reducer_weights=reducer_weights,
        learned_macs_per_sample_per_block=block_params,
        logical_boundary_bytes=batch * boundary_width * element_bytes,
        logical_local_bytes_if_materialized_per_block=(
            batch * local_width * element_bytes
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
        f"{'mode':<6} {'candidate':<15} {'Rref':>5} {'B':>5} {'out':>6} "
        f"{'H':>7} {'fwd_ms':>9} {'fb_ms':>9} {'peakF_MB':>10} {'peakFB_MB':>10}"
    )
    for r in rows:
        fb = "-" if r.forward_backward_ms is None else f"{r.forward_backward_ms:.4f}"
        peak_fb = (
            "-"
            if r.peak_forward_backward_active_bytes_delta is None
            else f"{r.peak_forward_backward_active_bytes_delta / 2**20:.3f}"
        )
        print(
            f"{r.mode:<6} {r.candidate:<15} {r.reference_receiver_width:>5} "
            f"{r.batch:>5} {r.boundary_width:>6} {r.local_width:>7} "
            f"{r.forward_ms:>9.4f} {fb:>9} "
            f"{r.peak_forward_active_bytes_delta / 2**20:>10.3f} {peak_fb:>10}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--candidates",
        nargs="+",
        default=[
            "wide_point",
            "dense",
            "fixed_sum",
            "grouped2",
            "grouped4",
            "lowrank16",
        ],
    )
    p.add_argument(
        "--modes",
        nargs="+",
        choices=["micro", "stack"],
        default=["micro"],
    )
    p.add_argument(
        "--receiver-widths",
        type=int,
        nargs="+",
        default=[256, 512, 1024],
        help="reference narrow R values; wide_point uses D=R*sqrt(K)",
    )
    p.add_argument("--budget-factor", type=int, default=16)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--batches", type=int, nargs="+", default=[32, 128, 512])
    p.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
    )
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
        args.candidates = ["wide_point", "dense", "fixed_sum"]
        args.modes = ["micro"]
        args.receiver_widths = [args.receiver_widths[0]]
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
        for receiver_width in args.receiver_widths:
            for candidate in args.candidates:
                for batch in args.batches:
                    rows.append(
                        bench_one(
                            mode=mode,
                            candidate=candidate,
                            receiver_width=receiver_width,
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
        print(
            "\nNOTE: active-allocation deltas are not DRAM traffic. "
            "The size sweep is designed to expose cache/working-set crossovers; "
            "confirm memory-access claims with hardware counters."
        )


if __name__ == "__main__":
    main()
