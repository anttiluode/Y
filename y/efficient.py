"""Gate-2 controls for communication-bounded computation.

These blocks deliberately use ordinary efficient-linear ideas. They are not
novelty claims and they are not biological models. Gate 2 asks whether any
structured/local reducer can preserve the dense bottleneck's accuracy while
changing *measured* hardware cost.

All blocks expose the same narrow receiver width R. For a hidden budget factor
K, the nominal hidden learned-weight budget is

    B = K * R * R.

The implementations keep logical accounting explicit. Parameter count,
logical receiver width, CUDA allocation, latency, and physical DRAM traffic
are different quantities and must not be conflated.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import torch
from torch import Tensor, nn

from .layers import BranchReduceLinear


FixedReduction = Literal["mean", "sqrt_sum", "sum"]


def hidden_budget(receiver_width: int, budget_factor: int) -> int:
    if receiver_width <= 0 or budget_factor <= 0:
        raise ValueError("receiver_width and budget_factor must be positive")
    return int(budget_factor) * int(receiver_width) * int(receiver_width)


class DenseBudgetBlock(nn.Module):
    """Ordinary dense expansion -> ReLU -> dense reduction control."""

    kind = "dense"

    def __init__(self, receiver_width: int, budget_factor: int) -> None:
        super().__init__()
        if (budget_factor * receiver_width) % 2:
            raise ValueError("budget_factor * receiver_width must be even")
        self.receiver_width = int(receiver_width)
        self.budget_factor = int(budget_factor)
        self.local_width = self.budget_factor * self.receiver_width // 2
        self.up = nn.Linear(self.receiver_width, self.local_width, bias=False)
        self.down = nn.Linear(self.local_width, self.receiver_width, bias=False)
        self.act = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.down(self.act(self.up(x)))

    @property
    def reducer_weights(self) -> int:
        return self.local_width * self.receiver_width

    @property
    def nominal_budget(self) -> int:
        return hidden_budget(self.receiver_width, self.budget_factor)


class LowRankReducerBlock(nn.Module):
    """Dense local feature generation with a low-rank learned reducer.

    The reducer H -> R is factorized H -> q -> R with no nonlinearity between
    factors, so it is genuinely a rank-q linear receiver. Local width H is
    chosen as large as possible without exceeding K*R^2 learned weights.

    Initialization is variance-matched to a single default ``nn.Linear(H,R)``.
    Two independently initialized linear factors otherwise make the effective
    H->R matrix about sqrt(3) smaller in standard deviation, which unfairly
    handicaps low rank in short training runs. Scaling the second factor by
    sqrt(3) makes the effective per-entry variance approximately 1/(3H), the
    same as PyTorch's default dense linear initialization.
    """

    kind = "lowrank"

    def __init__(self, receiver_width: int, budget_factor: int, rank: int) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.receiver_width = int(receiver_width)
        self.budget_factor = int(budget_factor)
        self.rank = int(rank)
        budget = hidden_budget(self.receiver_width, self.budget_factor)
        numerator = budget - self.rank * self.receiver_width
        denominator = self.receiver_width + self.rank
        if numerator <= 0:
            raise ValueError("rank consumes the entire hidden budget")
        self.local_width = numerator // denominator
        if self.local_width <= 0:
            raise ValueError("no local feature budget remains")
        self.up = nn.Linear(self.receiver_width, self.local_width, bias=False)
        self.reduce_in = nn.Linear(self.local_width, self.rank, bias=False)
        self.reduce_out = nn.Linear(self.rank, self.receiver_width, bias=False)
        with torch.no_grad():
            self.reduce_out.weight.mul_(math.sqrt(3.0))
        self.act = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        h = self.act(self.up(x))
        return self.reduce_out(self.reduce_in(h))

    @property
    def reducer_weights(self) -> int:
        return self.local_width * self.rank + self.rank * self.receiver_width

    @property
    def nominal_budget(self) -> int:
        return hidden_budget(self.receiver_width, self.budget_factor)


class GroupedReducerBlock(nn.Module):
    """Dense local feature generation with a block-diagonal learned receiver.

    ``groups=1`` is exactly the ordinary dense bottleneck endpoint. Increasing
    groups makes the H -> R receiver increasingly local: each output group can
    read only its corresponding local feature group. The saved reducer budget
    is spent on more private nonlinear features while total learned weights do
    not exceed K*R^2.

    This is a standard grouped/block-linear control, not a claimed Y primitive.
    """

    kind = "grouped"

    def __init__(self, receiver_width: int, budget_factor: int, groups: int) -> None:
        super().__init__()
        if groups <= 0:
            raise ValueError("groups must be positive")
        if receiver_width % groups:
            raise ValueError("receiver_width must be divisible by groups")
        self.receiver_width = int(receiver_width)
        self.budget_factor = int(budget_factor)
        self.groups = int(groups)
        budget = hidden_budget(self.receiver_width, self.budget_factor)

        # Cost = R*H for dense feature generation + R*H/G for grouped reducer.
        raw_width = (budget * self.groups) // (
            self.receiver_width * (self.groups + 1)
        )
        self.local_width = raw_width - (raw_width % self.groups)
        if self.local_width <= 0:
            raise ValueError("no local feature budget remains")

        self.local_per_group = self.local_width // self.groups
        self.receiver_per_group = self.receiver_width // self.groups
        self.up = nn.Linear(self.receiver_width, self.local_width, bias=False)
        self.weight = nn.Parameter(
            torch.empty(
                self.groups,
                self.receiver_per_group,
                self.local_per_group,
            )
        )
        self.act = nn.ReLU()
        # Treat each [receiver_per_group, local_per_group] slice like an
        # independent nn.Linear. Generic Kaiming init on this 3-D tensor would
        # count the receiver axis as fan-in and shrink the grouped control.
        bound = 1.0 / math.sqrt(self.local_per_group)
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        h = self.act(self.up(x))
        leading = h.shape[:-1]
        grouped = h.reshape(-1, self.groups, self.local_per_group)
        # [N,G,Hg] x [G,Rg,Hg] -> [N,G,Rg]
        y = torch.einsum("ngh,grh->ngr", grouped, self.weight)
        return y.reshape(*leading, self.receiver_width)

    @property
    def reducer_weights(self) -> int:
        return self.groups * self.receiver_per_group * self.local_per_group

    @property
    def nominal_budget(self) -> int:
        return hidden_budget(self.receiver_width, self.budget_factor)


class FixedBranchBudgetBlock(nn.Module):
    """Fixed local nonlinear aggregation endpoint.

    It spends the full learned hidden budget on R -> K*R feature generation and
    uses a parameter-free reducer. Wu et al.'s formal dendritic block **sums**
    K branch outputs. Earlier Y gates used a mean, which is representationally
    only a constant rescaling but can change optimization in a deep unnormalized
    MLP. Gate 2 therefore exposes all three scale controls:

    ``mean``      = sum / K       (historical Y endpoint)
    ``sqrt_sum``  = sum / sqrt(K) (variance-preserving scale audit)
    ``sum``       = raw paper-form aggregation

    They have identical parameters and connectivity. A difference between them
    is an optimization/normalization finding, not an architectural win.
    """

    kind = "fixed_branch"

    def __init__(
        self,
        receiver_width: int,
        budget_factor: int,
        *,
        reduction: FixedReduction = "mean",
    ) -> None:
        super().__init__()
        if reduction not in ("mean", "sqrt_sum", "sum"):
            raise ValueError("reduction must be mean, sqrt_sum, or sum")
        self.receiver_width = int(receiver_width)
        self.budget_factor = int(budget_factor)
        self.reduction: FixedReduction = reduction
        self.local_width = self.receiver_width * self.budget_factor
        self.up = nn.Linear(self.receiver_width, self.local_width, bias=False)
        self.act = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        h = self.act(self.up(x))
        shape = (*h.shape[:-1], self.receiver_width, self.budget_factor)
        grouped = h.reshape(shape)
        if self.reduction == "mean":
            return grouped.mean(dim=-1)
        result = grouped.sum(dim=-1)
        if self.reduction == "sqrt_sum":
            result = result / math.sqrt(self.budget_factor)
        return result

    @property
    def reducer_weights(self) -> int:
        return 0

    @property
    def nominal_budget(self) -> int:
        return hidden_budget(self.receiver_width, self.budget_factor)


def block_parameter_count(block: nn.Module) -> int:
    return sum(p.numel() for p in block.parameters())


def block_budget_slack(block: nn.Module) -> int:
    nominal = int(getattr(block, "nominal_budget"))
    return nominal - block_parameter_count(block)


class EfficientControlMLP(nn.Module):
    """Matched narrow MLP for Gate-2 accuracy experiments.

    All variants share the same input stem and output head. Only the hidden
    receiver-to-receiver block changes.
    """

    def __init__(
        self,
        input_dim: int,
        receiver_width: int,
        depth: int,
        budget_factor: int,
        classes: int,
        block_factory: Callable[[], nn.Module],
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        root = int(math.isqrt(budget_factor))
        if root * root != budget_factor:
            raise ValueError("budget_factor must be a perfect square for matched stem")
        self.stem = BranchReduceLinear(
            input_dim,
            receiver_width,
            branches=root,
            reduction="mean",
            bias=False,
        )
        self.blocks = nn.Sequential(*(block_factory() for _ in range(depth - 1)))
        self.head = nn.Linear(receiver_width, classes, bias=False)
        self.receiver_width = int(receiver_width)
        self.depth = int(depth)
        self.budget_factor = int(budget_factor)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.blocks(self.stem(x)))


def make_control_mlp(
    kind: str,
    *,
    input_dim: int,
    receiver_width: int,
    depth: int,
    budget_factor: int,
    classes: int,
    rank: int | None = None,
    groups: int | None = None,
    fixed_reduction: FixedReduction = "mean",
) -> EfficientControlMLP:
    kind = kind.lower()
    if kind == "dense":
        factory = lambda: DenseBudgetBlock(receiver_width, budget_factor)
    elif kind == "fixed":
        factory = lambda: FixedBranchBudgetBlock(
            receiver_width, budget_factor, reduction=fixed_reduction
        )
    elif kind == "lowrank":
        if rank is None:
            raise ValueError("lowrank requires rank")
        factory = lambda: LowRankReducerBlock(receiver_width, budget_factor, rank)
    elif kind == "grouped":
        if groups is None:
            raise ValueError("grouped requires groups")
        factory = lambda: GroupedReducerBlock(receiver_width, budget_factor, groups)
    else:
        raise ValueError(f"unknown control kind: {kind}")
    return EfficientControlMLP(
        input_dim,
        receiver_width,
        depth,
        budget_factor,
        classes,
        factory,
    )
