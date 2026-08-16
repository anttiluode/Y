"""Small, explicit accounting helpers for communication-bounded blocks.

These are *logical traffic proxies*, not hardware profiler measurements.
The repo keeps those two claims separate on purpose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BoundaryTraffic:
    batch: int
    width: int
    boundaries: int = 1
    bytes_per_value: int = 2

    @property
    def values(self) -> int:
        return self.batch * self.width * self.boundaries

    @property
    def bytes(self) -> int:
        return self.values * self.bytes_per_value


def matched_receiver_width(point_width: int, branches: int) -> int:
    """Width for a square branch-reduce layer at matched weight complexity.

    A point layer with width D has D^2 weights. A branch-reduce layer with
    receiver width R and K branches has K R^2 weights. Matching gives

        R = D / sqrt(K).

    We return the nearest positive integer. Callers should report the exact
    parameter count as well, because rounding breaks exact equality.
    """
    if point_width <= 0 or branches <= 0:
        raise ValueError("point_width and branches must be positive")
    return max(1, int(round(point_width / math.sqrt(branches))))


def square_weight_count(width: int) -> int:
    if width <= 0:
        raise ValueError("width must be positive")
    return width * width


def branch_square_weight_count(receiver_width: int, branches: int) -> int:
    if receiver_width <= 0 or branches <= 0:
        raise ValueError("receiver_width and branches must be positive")
    return branches * receiver_width * receiver_width


def receiver_traffic_ratio(point_width: int, receiver_width: int) -> float:
    if point_width <= 0 or receiver_width <= 0:
        raise ValueError("widths must be positive")
    return receiver_width / point_width
