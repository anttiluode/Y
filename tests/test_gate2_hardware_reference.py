import torch

from experiments.gate2_hardware_shortlist import (
    WidePointBudgetBlock,
    boundary_width_for,
    make_core,
)
from y.efficient import (
    DenseBudgetBlock,
    FixedBranchBudgetBlock,
    block_parameter_count,
    hidden_budget,
)


def test_paper_axis_has_exact_equal_learned_budget() -> None:
    r = 32
    k = 16
    wide = WidePointBudgetBlock(r, k)
    fixed = FixedBranchBudgetBlock(r, k, reduction="sum")
    dense = DenseBudgetBlock(r, k)
    budget = hidden_budget(r, k)

    assert block_parameter_count(wide) == budget
    assert block_parameter_count(fixed) == budget
    assert block_parameter_count(dense) == budget


def test_paper_axis_boundary_ratio_is_sqrt_k() -> None:
    r = 32
    k = 16
    assert boundary_width_for("wide_point", r, k) == 128
    assert boundary_width_for("fixed_sum", r, k) == 32
    assert boundary_width_for("dense", r, k) == 32


def test_wide_point_requires_square_budget_factor() -> None:
    try:
        WidePointBudgetBlock(32, 8)
    except ValueError:
        pass
    else:
        raise AssertionError("non-square K must fail exact wide-point matching")


def test_scale_controlled_stack_shapes_match_each_candidate_boundary() -> None:
    wide_core, _wide_block, depth, wide_boundary = make_core(
        mode="stack",
        candidate="wide_point",
        receiver_width=16,
        budget_factor=4,
        depth=3,
    )
    fixed_core, _fixed_block, fixed_depth, fixed_boundary = make_core(
        mode="stack",
        candidate="fixed_sum",
        receiver_width=16,
        budget_factor=4,
        depth=3,
    )

    assert depth == fixed_depth == 3
    assert wide_boundary == 32
    assert fixed_boundary == 16
    assert wide_core(torch.randn(2, 32)).shape == (2, 32)
    assert fixed_core(torch.randn(2, 16)).shape == (2, 16)
