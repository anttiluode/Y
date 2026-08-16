"""Tiny reference models used by the first falsification gate."""

from __future__ import annotations

import math

from torch import Tensor, nn

from .layers import BranchReduceLinear


class PointMLP(nn.Module):
    def __init__(self, input_dim: int, width: int, depth: int, classes: int) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        layers: list[nn.Module] = [nn.Linear(input_dim, width, bias=False), nn.ReLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width, bias=False), nn.ReLU()])
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(width, classes, bias=False)
        self.receiver_width = width
        self.depth = depth

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.body(x))


class BranchMLP(nn.Module):
    """Deep network whose block-to-block interface is a narrow receiver."""

    def __init__(self, input_dim: int, receiver_width: int, depth: int, branches: int, classes: int) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        root = int(math.isqrt(branches))
        if root * root != branches:
            raise ValueError("BranchMLP currently requires a perfect-square branch count")
        layers: list[nn.Module] = [
            BranchReduceLinear(input_dim, receiver_width, root, reduction="mean", bias=False)
        ]
        for _ in range(depth - 1):
            layers.append(
                BranchReduceLinear(receiver_width, receiver_width, branches, reduction="mean", bias=False)
            )
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(receiver_width, classes, bias=False)
        self.receiver_width = receiver_width
        self.branches = branches
        self.input_branches = root
        self.depth = depth

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.body(x))


class NarrowBottleneckBlock(nn.Module):
    """Ordinary learned local expansion/compression at a narrow interface.

    With receiver width R, choose local_width so the two matrices have the
    same weight count as a K-branch BranchReduceLinear:

        2 * R * local_width = K * R^2.
    """

    def __init__(self, receiver_width: int, local_width: int) -> None:
        super().__init__()
        self.up = nn.Linear(receiver_width, local_width, bias=False)
        self.act = nn.ReLU()
        self.down = nn.Linear(local_width, receiver_width, bias=False)
        self.receiver_width = receiver_width
        self.local_width = local_width

    def forward(self, x: Tensor) -> Tensor:
        return self.down(self.act(self.up(x)))


class BottleneckMLP(nn.Module):
    """Same narrow interface and hidden weight budget as BranchMLP, but ordinary MLP compression."""

    def __init__(self, input_dim: int, receiver_width: int, depth: int, branches: int, classes: int) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        root = int(math.isqrt(branches))
        if root * root != branches:
            raise ValueError("BottleneckMLP currently requires a perfect-square branch count")
        if (branches * receiver_width) % 2 != 0:
            raise ValueError("branches * receiver_width must be even")
        local_width = branches * receiver_width // 2
        layers: list[nn.Module] = [
            BranchReduceLinear(input_dim, receiver_width, root, reduction="mean", bias=False)
        ]
        for _ in range(depth - 1):
            layers.append(NarrowBottleneckBlock(receiver_width, local_width))
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(receiver_width, classes, bias=False)
        self.receiver_width = receiver_width
        self.local_width = local_width
        self.branches_equivalent = branches
        self.depth = depth

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.body(x))


class MixedBranchBlock(nn.Module):
    """Negative control: branch collapse followed by a small learned mixer.

    For budget factor K and receiver width R, use K-1 branch groups followed
    by an R x R mixer, preserving K*R^2 hidden weights. Because the mixer sees
    only the already-reduced R-vector, it cannot directly act on expanded local
    branch state. In a feed-forward stack much of this extra linear map can
    also be absorbed into the next layer, so this is expected to be a weak use
    of the parameter budget. It is kept as a tempting design that failed.
    """

    def __init__(self, receiver_width: int, budget_factor: int) -> None:
        super().__init__()
        if budget_factor < 2:
            raise ValueError("budget_factor must be >= 2")
        self.branch = BranchReduceLinear(
            receiver_width,
            receiver_width,
            branches=budget_factor - 1,
            reduction="mean",
            bias=False,
        )
        self.mix = nn.Linear(receiver_width, receiver_width, bias=False)
        self.receiver_width = receiver_width
        self.local_branches = budget_factor - 1
        self.budget_factor = budget_factor

    def forward(self, x: Tensor) -> Tensor:
        return self.mix(self.branch(x))


class MixedBranchMLP(nn.Module):
    """Network using the post-collapse mixer negative-control block."""

    def __init__(self, input_dim: int, receiver_width: int, depth: int, branches: int, classes: int) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        root = int(math.isqrt(branches))
        if root * root != branches:
            raise ValueError("MixedBranchMLP currently requires a perfect-square branch count")
        layers: list[nn.Module] = [
            BranchReduceLinear(input_dim, receiver_width, root, reduction="mean", bias=False)
        ]
        for _ in range(depth - 1):
            layers.append(MixedBranchBlock(receiver_width, branches))
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(receiver_width, classes, bias=False)
        self.receiver_width = receiver_width
        self.branches_equivalent = branches
        self.depth = depth

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.body(x))
