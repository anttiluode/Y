"""Y: experiments in communication-bounded neural computation."""

from .accounting import (
    BoundaryTraffic,
    branch_square_weight_count,
    matched_receiver_width,
    receiver_traffic_ratio,
    square_weight_count,
)
from .layers import BranchReduceLinear
from .models import (
    BottleneckMLP,
    BranchMLP,
    MixedBranchBlock,
    MixedBranchMLP,
    NarrowBottleneckBlock,
    PointMLP,
    PrecollapseReceiverBlock,
    PrecollapseReceiverMLP,
)

__all__ = [
    "BoundaryTraffic",
    "BranchReduceLinear",
    "BranchMLP",
    "BottleneckMLP",
    "NarrowBottleneckBlock",
    "MixedBranchBlock",
    "MixedBranchMLP",
    "PrecollapseReceiverBlock",
    "PrecollapseReceiverMLP",
    "PointMLP",
    "branch_square_weight_count",
    "matched_receiver_width",
    "receiver_traffic_ratio",
    "square_weight_count",
]
