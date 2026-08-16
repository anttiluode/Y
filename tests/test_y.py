import torch
from torch import nn

from y import (
    BoundaryTraffic,
    BottleneckMLP,
    BranchMLP,
    BranchReduceLinear,
    MixedBranchMLP,
    PointMLP,
    PrecollapseReceiverBlock,
    PrecollapseReceiverMLP,
    branch_square_weight_count,
    matched_receiver_width,
    receiver_traffic_ratio,
    square_weight_count,
)


def nparams(model) -> int:
    return sum(p.numel() for p in model.parameters())


def test_branch_reduce_shape_and_mean() -> None:
    layer = BranchReduceLinear(3, 2, branches=4, activation=nn.Identity(), reduction="mean", bias=False)
    with torch.no_grad():
        layer.proj.weight.fill_(1.0)
    x = torch.tensor([[1.0, 2.0, 3.0]])
    branches = layer.branch_state(x)
    y = layer(x)
    assert branches.shape == (1, 2, 4)
    assert y.shape == (1, 2)
    assert torch.allclose(y, torch.tensor([[6.0, 6.0]]))


def test_sum_is_mean_times_branches() -> None:
    x = torch.randn(5, 7)
    a = BranchReduceLinear(7, 3, branches=4, reduction="mean")
    b = BranchReduceLinear(7, 3, branches=4, reduction="sum")
    b.load_state_dict(a.state_dict())
    assert torch.allclose(b(x), a(x) * 4, atol=1e-6)


def test_exact_square_matching_for_square_branch_counts() -> None:
    assert matched_receiver_width(128, 4) == 64
    assert matched_receiver_width(128, 16) == 32
    assert square_weight_count(128) == branch_square_weight_count(64, 4)
    assert square_weight_count(128) == branch_square_weight_count(32, 16)


def test_receiver_ratio() -> None:
    assert receiver_traffic_ratio(128, 64) == 0.5
    assert receiver_traffic_ratio(128, 32) == 0.25


def test_boundary_bytes() -> None:
    t = BoundaryTraffic(batch=8, width=64, boundaries=3, bytes_per_value=2)
    assert t.values == 8 * 64 * 3
    assert t.bytes == 8 * 64 * 3 * 2


def test_complexity_matching_is_close_for_gate0_shapes() -> None:
    point = PointMLP(input_dim=32, width=128, depth=4, classes=8)
    k4 = BranchMLP(input_dim=32, receiver_width=64, depth=4, branches=4, classes=8)
    k16 = BranchMLP(input_dim=32, receiver_width=32, depth=4, branches=16, classes=8)
    assert nparams(k4) / nparams(point) > 0.98
    assert nparams(k16) / nparams(point) > 0.98
    assert k4.input_branches == 2
    assert k16.input_branches == 4


def test_bottleneck_control_matches_branch_budget() -> None:
    branch = BranchMLP(input_dim=32, receiver_width=64, depth=4, branches=4, classes=8)
    bneck = BottleneckMLP(input_dim=32, receiver_width=64, depth=4, branches=4, classes=8)
    assert nparams(branch) == nparams(bneck)
    assert bneck.local_width == 128


def test_postcollapse_mixer_matches_hidden_budget() -> None:
    branch = BranchMLP(input_dim=32, receiver_width=32, depth=4, branches=16, classes=8)
    mixed = MixedBranchMLP(input_dim=32, receiver_width=32, depth=4, branches=16, classes=8)
    assert nparams(branch) == nparams(mixed)


def test_precollapse_zero_fanin_matches_branch_reduce() -> None:
    branch = BranchReduceLinear(8, 8, branches=4, reduction="mean", bias=False)
    pre = PrecollapseReceiverBlock(8, budget_factor=4, reducer_fanin=0)
    with torch.no_grad():
        pre.up.weight.copy_(branch.proj.weight)
    x = torch.randn(6, 8)
    assert torch.allclose(branch(x), pre(x), atol=1e-6)


def test_precollapse_frontier_keeps_exact_hidden_budget() -> None:
    receiver_width = 32
    budget_factor = 16
    target = budget_factor * receiver_width * receiver_width
    for fanin in (0, 32, 64, 128, 256):
        block = PrecollapseReceiverBlock(
            receiver_width,
            budget_factor,
            reducer_fanin=fanin,
            index_seed=7,
        )
        assert nparams(block) == target
        assert block.hidden_weight_budget == target
        assert block.learned_reducer_weights == receiver_width * fanin
        assert block.correction_index.shape == (receiver_width, fanin)


def test_precollapse_mlp_matches_branch_and_bottleneck_parameter_count() -> None:
    branch = BranchMLP(input_dim=64, receiver_width=32, depth=4, branches=16, classes=10)
    bneck = BottleneckMLP(input_dim=64, receiver_width=32, depth=4, branches=16, classes=10)
    for fanin in (0, 32, 64, 128, 256):
        pre = PrecollapseReceiverMLP(
            input_dim=64,
            receiver_width=32,
            depth=4,
            branches=16,
            reducer_fanin=fanin,
            classes=10,
            index_seed=11,
        )
        assert nparams(pre) == nparams(branch) == nparams(bneck)
