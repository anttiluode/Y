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


def test_fixed_branch_endpoint_uses_exact_budget() -> None:
    block = FixedBranchBudgetBlock(32, 16)
    assert block.local_width == 512
    assert block.reducer_weights == 0
    assert block_parameter_count(block) == hidden_budget(32, 16)


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
