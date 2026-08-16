"""Communication-bounded neural building blocks.

The first layer in Y intentionally mirrors the core abstraction in
Wu et al. (2026): several nonlinear branch outputs are aggregated locally
before a narrower receiver vector is exposed to the next block.

This file contains no claim of biological fidelity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from torch import Tensor, nn

Reduction = Literal["sum", "mean"]


class BranchReduceLinear(nn.Module):
    """Linear branches with local nonlinearity and a narrow shared receiver.

    For input x with shape [..., in_features], compute

        b = activation(W x + bias)      # [..., out_features, branches]
        y = reduce_k b[..., k]          # [..., out_features]

    The branch tensor is local to this module. Only ``y`` is part of the
    public interface. A fused implementation may avoid materializing ``b``
    in global memory; this reference implementation favors clarity.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        branches: int = 4,
        *,
        activation: nn.Module | None = None,
        reduction: Reduction = "mean",
        bias: bool = True,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        if branches <= 0:
            raise ValueError("branches must be positive")
        if reduction not in ("sum", "mean"):
            raise ValueError("reduction must be 'sum' or 'mean'")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.branches = int(branches)
        self.reduction: Reduction = reduction
        self.proj = nn.Linear(
            self.in_features,
            self.out_features * self.branches,
            bias=bias,
        )
        self.activation = activation if activation is not None else nn.ReLU()

    def branch_state(self, x: Tensor) -> Tensor:
        """Return the local branch state as [..., out_features, branches]."""
        shape = (*x.shape[:-1], self.out_features, self.branches)
        return self.activation(self.proj(x)).reshape(shape)

    def forward(self, x: Tensor) -> Tensor:
        branches = self.branch_state(x)
        if self.reduction == "sum":
            return branches.sum(dim=-1)
        return branches.mean(dim=-1)

    @property
    def receiver_width(self) -> int:
        return self.out_features

    @property
    def local_branch_width(self) -> int:
        return self.out_features * self.branches


@dataclass(frozen=True)
class BranchShape:
    receiver_width: int
    branches: int

    @property
    def local_width(self) -> int:
        return self.receiver_width * self.branches
