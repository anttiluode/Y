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
    """Deep network whose block-to-block interface is a narrow receiver.

    For perfect-square branch counts K, the first layer uses sqrt(K) branches.
    If receiver width R = D/sqrt(K), this makes its input-layer weight count
    match a point model's input_dim*D weights, while hidden square blocks use K
    branches and match D^2 weights. This mirrors the complexity-matching logic
    used for fixed input/output dimensionality in the reference paper.
    """

    def __init__(
        self,
        input_dim: int,
        receiver_width: int,
        depth: int,
        branches: int,
        classes: int,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        root = int(math.isqrt(branches))
        if root * root != branches:
            raise ValueError("BranchMLP currently requires a perfect-square branch count")

        layers: list[nn.Module] = [
            BranchReduceLinear(
                input_dim,
                receiver_width,
                root,
                reduction="mean",
                bias=False,
            )
        ]
        for _ in range(depth - 1):
            layers.append(
                BranchReduceLinear(
                    receiver_width,
                    receiver_width,
                    branches,
                    reduction="mean",
                    bias=False,
                )
            )
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(receiver_width, classes, bias=False)
        self.receiver_width = receiver_width
        self.branches = branches
        self.input_branches = root
        self.depth = depth

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.body(x))
