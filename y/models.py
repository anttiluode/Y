"""Tiny reference models used by the first falsification gates."""

from __future__ import annotations

import math

import torch
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


class PrecollapseReceiverBlock(nn.Module):
    """Fixed-budget local feature bank with a learned sparse pre-collapse receiver.

    Let R be the communicated receiver width and K the hidden budget factor.
    A branch block spends K*R^2 learned weights on local feature generation and
    uses a parameter-free grouped mean to expose R values. A dense bottleneck
    spends half that budget on local features and half on a learned reducer.

    This block interpolates between those ideas without changing the learned
    hidden budget. If each receiver has ``s`` learned correction edges, choose

        H = K*R - s

    local nonlinear features. Then

        R*H + R*s = K*R^2

    exactly. All H features still have a parameter-free grouped-mean path to
    the receiver. The sparse learned correction acts on the *pre-collapse*
    local state and is zero-initialized, so optimization begins at the fixed
    grouped receiver and can learn departures from it.

    This is a reference implementation. The gathered sparse correction is not
    claimed to be hardware-efficient until a fused/local kernel is measured.
    """

    def __init__(
        self,
        receiver_width: int,
        budget_factor: int,
        reducer_fanin: int,
        *,
        index_seed: int = 0,
    ) -> None:
        super().__init__()
        if receiver_width <= 0:
            raise ValueError("receiver_width must be positive")
        if budget_factor <= 0:
            raise ValueError("budget_factor must be positive")
        if reducer_fanin < 0:
            raise ValueError("reducer_fanin must be non-negative")

        local_width = budget_factor * receiver_width - reducer_fanin
        if local_width <= 0:
            raise ValueError("reducer_fanin leaves no local feature budget")
        if reducer_fanin > local_width:
            raise ValueError("reducer_fanin cannot exceed local_width")
        if local_width % receiver_width != 0:
            raise ValueError(
                "local_width must divide evenly into receiver groups; "
                "choose reducer_fanin so (K*R - s) % R == 0"
            )

        self.receiver_width = int(receiver_width)
        self.budget_factor = int(budget_factor)
        self.reducer_fanin = int(reducer_fanin)
        self.local_width = int(local_width)
        self.group_size = self.local_width // self.receiver_width
        self.up = nn.Linear(self.receiver_width, self.local_width, bias=False)

        if self.reducer_fanin == 0:
            self.register_buffer(
                "correction_index",
                torch.empty(self.receiver_width, 0, dtype=torch.long),
            )
            self.register_parameter("correction_weight", None)
        else:
            g = torch.Generator().manual_seed(
                int(index_seed)
                + self.receiver_width * 97
                + self.budget_factor * 991
                + self.reducer_fanin * 7919
            )
            if self.reducer_fanin == self.local_width:
                correction_index = torch.arange(self.local_width).repeat(
                    self.receiver_width, 1
                )
            else:
                correction_index = torch.stack(
                    [
                        torch.randperm(self.local_width, generator=g)[
                            : self.reducer_fanin
                        ]
                        for _ in range(self.receiver_width)
                    ]
                )
            self.register_buffer("correction_index", correction_index)
            self.correction_weight = nn.Parameter(
                torch.zeros(self.receiver_width, self.reducer_fanin)
            )

    def local_state(self, x: Tensor) -> Tensor:
        return torch.relu(self.up(x))

    def base_receiver(self, h: Tensor) -> Tensor:
        shape = (*h.shape[:-1], self.receiver_width, self.group_size)
        return h.reshape(shape).mean(dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        h = self.local_state(x)
        receiver = self.base_receiver(h)
        if self.reducer_fanin == 0:
            return receiver
        gathered = h[..., self.correction_index]
        correction = (gathered * self.correction_weight).sum(dim=-1)
        return receiver + correction

    @property
    def learned_reducer_weights(self) -> int:
        return self.receiver_width * self.reducer_fanin

    @property
    def hidden_weight_budget(self) -> int:
        return self.budget_factor * self.receiver_width * self.receiver_width

    @property
    def reducer_budget_fraction(self) -> float:
        return self.reducer_fanin / (self.budget_factor * self.receiver_width)


class PrecollapseReceiverMLP(nn.Module):
    """Deep narrow network that sweeps the pre-collapse receiver budget."""

    def __init__(
        self,
        input_dim: int,
        receiver_width: int,
        depth: int,
        branches: int,
        reducer_fanin: int,
        classes: int,
        *,
        index_seed: int = 0,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        root = int(math.isqrt(branches))
        if root * root != branches:
            raise ValueError(
                "PrecollapseReceiverMLP currently requires a perfect-square branch count"
            )

        layers: list[nn.Module] = [
            BranchReduceLinear(
                input_dim,
                receiver_width,
                root,
                reduction="mean",
                bias=False,
            )
        ]
        for i in range(depth - 1):
            layers.append(
                PrecollapseReceiverBlock(
                    receiver_width,
                    branches,
                    reducer_fanin,
                    index_seed=index_seed + i * 101,
                )
            )
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(receiver_width, classes, bias=False)
        self.receiver_width = receiver_width
        self.branches_equivalent = branches
        self.reducer_fanin = reducer_fanin
        self.local_width = branches * receiver_width - reducer_fanin
        self.reducer_budget_fraction = reducer_fanin / (branches * receiver_width)
        self.depth = depth

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.body(x))
