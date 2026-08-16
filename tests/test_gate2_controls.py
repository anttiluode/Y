import math

import pytest
import torch

from y.efficient import (
    DenseBudgetBlock,
    FixedBranchBudgetBlock,
    GroupedReducerBlock,
    LowRankReducerBlock,
    block_budget_slack,
    block_parameter_count,
    hidden_budget,
    make_control_mlp,
)


def test_dense_endpoint_uses_exact_budget() -> None:
    block = DenseBudgetBlock(32, 16)
    assert block.local_width == 256
    assert block_parameter_count(block) == hidden_budget(32, 16)
    assert block_budget_slack(block) == 0


def test_grouped_one_is_dense_budget_endpoint() -> None:
    dense = DenseBudgetBlock(32, 16)
    grouped = GroupedReducerBlock(32, 16, groups=1)
    assert grouped.local_width == dense.local_width
    assert block_parameter_count(grouped) == block_parameter_count(dense)
    assert grouped.reducer_weights == dense.reducer_weights


def test_grouped_reducer_trades_reducer_weights_for_local_width() -> None:
    dense = DenseBudgetBlock(32, 16)
    previous_width = dense.local_width
    previous_reducer = dense.reducer_weights
    for groups in (2, 4, 8, 16, 32):
        block = GroupedReducerBlock(32, 16, groups=groups)
        assert block.local_width >= previous_width
        assert block.reducer_weights <= previous_reducer
        assert block_parameter_count(block) <= hidden_budget(32, 16)
        assert block_budget_slack(block) >= 0
        previous_width = block.local_width
        previous_reducer = block.reducer_weights


def test_lowrank_reducer_stays_within_budget() -> None:
    for rank in (2, 4, 8, 16, 32):
        block = LowRankReducerBlock(32, 16, rank=rank)
        assert block_parameter_count(block) <= hidden_budget(32, 16)
        assert block_budget_slack(block) >= 0


def test_lowrank_second_factor_has_variance_matching_scale() -> None:
    # PyTorch's default Linear(q,R) bound is 1/sqrt(q). Gate 2 multiplies the
    # second factor by sqrt(3), making its variance 1/q so the product of the
    # two low-rank factors has the same approximate entry variance as one
    # default dense H->R linear map.
    torch.manual_seed(123)
    block = LowRankReducerBlock(64, 16, rank=32)
    max_expected = math.sqrt(3.0 / 32.0)
    assert float(block.reduce_out.weight.abs().max()) <= max_expected + 1e-6
    assert float(block.reduce_out.weight.std()) > 0.15


def test_fixed_branch_endpoint_uses_exact_budget() -> None:
    for reduction in ("mean", "sqrt_sum", "sum"):
        block = FixedBranchBudgetBlock(32, 16, reduction=reduction)
        assert block.local_width == 512
        assert block.reducer_weights == 0
        assert block_parameter_count(block) == hidden_budget(32, 16)


def test_fixed_branch_reductions_are_only_scale_changes() -> None:
    mean = FixedBranchBudgetBlock(8, 4, reduction="mean")
    sqrt_sum = FixedBranchBudgetBlock(8, 4, reduction="sqrt_sum")
    total = FixedBranchBudgetBlock(8, 4, reduction="sum")
    sqrt_sum.load_state_dict(mean.state_dict())
    total.load_state_dict(mean.state_dict())
    x = torch.randn(6, 8)
    y_mean = mean(x)
    assert torch.allclose(sqrt_sum(x), y_mean * math.sqrt(4.0), atol=1e-6)
    assert torch.allclose(total(x), y_mean * 4.0, atol=1e-6)


@pytest.mark.parametrize(
    "block",
    [
        DenseBudgetBlock(32, 16),
        GroupedReducerBlock(32, 16, groups=4),
        LowRankReducerBlock(32, 16, rank=8),
        FixedBranchBudgetBlock(32, 16),
    ],
)
def test_gate2_block_shapes(block) -> None:
    x = torch.randn(7, 32)
    y = block(x)
    assert y.shape == (7, 32)


def test_grouped_requires_divisible_receiver_width() -> None:
    with pytest.raises(ValueError):
        GroupedReducerBlock(30, 16, groups=8)


def test_make_control_mlp_shape() -> None:
    model = make_control_mlp(
        "grouped",
        input_dim=64,
        receiver_width=32,
        depth=4,
        budget_factor=16,
        classes=10,
        groups=4,
    )
    assert model(torch.randn(5, 64)).shape == (5, 10)


def test_make_fixed_sum_mlp_shape() -> None:
    model = make_control_mlp(
        "fixed",
        input_dim=64,
        receiver_width=32,
        depth=4,
        budget_factor=16,
        classes=10,
        fixed_reduction="sum",
    )
    assert model(torch.randn(5, 64)).shape == (5, 10)
