# Gate 0 — first receipt

**Date:** 2026-08-16  
**Status:** narrow receiver idea survives; fixed branch reduction is not privileged by the first control.

This is a synthetic instrument check, not an external benchmark and not a hardware-efficiency result.

## Question

At approximately matched parameter count, can rich local nonlinear computation preserve held-out classification performance while reducing the width communicated between hidden blocks?

The point model uses width `D=128`. Narrow models use receiver width `R` with the square hidden-core weight budget matched by

```text
K * R^2 ~= D^2
```

so:

```text
K=4   -> R=64   -> 0.50x logical receiver traffic
K=16  -> R=32   -> 0.25x logical receiver traffic
```

For the fixed input boundary, narrow models use `sqrt(K)` branch aggregation so the first-layer weight count is matched rather than accidentally over-allocated.

Two narrow internal blocks are compared at exactly the same hidden weight budget and receiver width:

```text
BRANCH
    Linear(R -> K*R) -> ReLU -> fixed branch mean -> R

BOTTLENECK CONTROL
    Linear(R -> K*R/2) -> ReLU -> learned Linear(K*R/2 -> R)
```

Both use `K*R^2` hidden weights. The bottleneck is the mandatory ordinary-MLP control: if it matches or beats the branch layer, the useful result is communication-bounded local compute, not something specifically dendritic.

## Setup

```text
input_dim       32
classes          8
hidden depth     4
point width    128
train samples 8192
test samples  2048
epochs           30
optimizer       AdamW
seeds          0,1,2
```

Labels come from a frozen random nonlinear teacher. Students do not see teacher weights.

The run used the development container with `torch 2.10.0+cpu`; timing is only reference-PyTorch timing, not a GPU traffic measurement.

## Results

| seed | point | branch K=4 | bottleneck K=4 | branch K=16 | bottleneck K=16 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.7983 | 0.7954 | 0.7720 | 0.7002 | 0.7739 |
| 1 | 0.7402 | 0.7236 | 0.7134 | 0.6631 | 0.6973 |
| 2 | 0.7495 | 0.7188 | 0.7271 | 0.6143 | 0.6992 |
| **mean** | **0.7627** | **0.7459** | **0.7375** | **0.6592** | **0.7235** |

Parameter counts:

```text
point                 54,272
branch K=4            53,760
bottleneck K=4        53,760
branch K=16           53,504
bottleneck K=16       53,504
```

Hidden square-block weights are exactly `16,384` in all cases.

Mean deltas versus point:

```text
branch K=4         -0.0167   at 0.50x receiver width
bottleneck K=4     -0.0252   at 0.50x receiver width
branch K=16        -0.1035   at 0.25x receiver width
bottleneck K=16    -0.0392   at 0.25x receiver width
```

## What changed after the control

The first branch-only run made `K=16` look like evidence that 4x interface compression was simply too aggressive. The ordinary bottleneck control falsifies that interpretation.

At the same quarter-width receiver and the same hidden weight count, the learned bottleneck recovers much of the lost accuracy:

```text
branch K=16 mean       0.6592
bottleneck K=16 mean   0.7235
```

Therefore the first earned statement is **not** “dendritic branches enable quarter-width communication.” It is closer to:

> A large amount of local computation can live behind a substantially narrower interface, but the way local state is compressed matters. Fixed branch averaging becomes a serious bottleneck at high compression in this toy; an ordinary learned bottleneck is much more robust there.

At `K=4`, branch reduction is slightly better than the bottleneck control on the three-seed mean, but the gap is small and synthetic. It is not evidence of a special dendritic advantage.

## Hardware note

All narrow reference implementations are slower than the point baseline in this CPU experiment because they execute extra local operations and materialize intermediate state. **Logical receiver width is not hardware efficiency.**

The Wu et al. public implementation already contains a fused Triton branch-matmul -> ReLU -> branch-reduction kernel. Y should not claim a hardware win until a comparable fused/local implementation is profiled on the same workload.

## Decision

**Survives:** communication-bounded local computation is worth pursuing.  
**Not earned:** fixed dendritic branch pooling as the privileged mechanism.  
**Killed:** the idea that the branch K=16 failure proves quarter-width receivers are intrinsically too small.

Next gate:

1. move both branch and ordinary bottleneck controls to a standard external dataset;
2. add an equal-traffic low-rank/bottleneck baseline before any receiver-routing story;
3. profile an actually fused/local implementation on GPU;
4. only then test whether a context-dependent receiver can beat one universal learned bottleneck at the same communicated width.

Do not tune this synthetic teacher to manufacture a branch win.
