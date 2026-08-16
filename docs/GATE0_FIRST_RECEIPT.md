# Gate 0 — first receipt

**Date:** 2026-08-16  
**Status:** narrow receiver idea survives; fixed branch reduction is not privileged; post-collapse mixing is a recorded null.

This is a synthetic instrument check, not an external benchmark and not a hardware-efficiency result.

## Question

At approximately matched parameter count, can rich local nonlinear computation preserve held-out classification performance while reducing the width communicated between hidden blocks?

The point model uses width `D=128`. Narrow models use receiver width `R` with square hidden-core weight budget

```text
K * R^2 ~= D^2
```

so `K=4 -> R=64` gives `0.50x` logical receiver width and `K=16 -> R=32` gives `0.25x`.

Three narrow mechanisms are compared at the **same receiver width and same hidden weight count**:

```text
BRANCH
    many local ReLU features -> fixed group mean -> R

BOTTLENECK CONTROL
    fewer local ReLU features -> learned pre-collapse compression -> R

POST-COLLAPSE MIXER NULL
    local ReLU branches -> fixed mean -> R -> learned R x R mixer
```

The third design is intentionally tempting: spend one branch-worth of budget on a small learned receiver mixer. But the mixer only sees the already-collapsed receiver, and in a feed-forward stack much of that extra linear map can be absorbed into the next layer. It is therefore a useful negative control for *where* learned compression must act.

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

Labels come from a frozen random nonlinear teacher. Students do not see teacher weights. Run environment: `torch 2.10.0+cpu`.

## Results

| model | receiver ratio | params | mean accuracy |
|---|---:|---:|---:|
| point D=128 | 1.00x | 54,272 | **0.7627** |
| branch K=4, R=64 | 0.50x | 53,760 | **0.7459** |
| bottleneck K=4, R=64 | 0.50x | 53,760 | **0.7375** |
| postmix K=4, R=64 | 0.50x | 53,760 | **0.7223** |
| branch K=16, R=32 | 0.25x | 53,504 | **0.6592** |
| bottleneck K=16, R=32 | 0.25x | 53,504 | **0.7235** |
| postmix K=16, R=32 | 0.25x | 53,504 | **0.4129** |

Per-seed post-collapse mixer results:

```text
K=4:   0.7739, 0.6904, 0.7026
K=16:  0.5308, 0.2681, 0.4399
```

Hidden square-block weights are exactly `16,384` in every model.

## What the controls changed

### 1. Quarter-width is not intrinsically impossible

The original branch-only result made `K=16` look like evidence that 4x interface compression was simply too aggressive. The ordinary bottleneck falsifies that interpretation:

```text
branch K=16 mean       0.6592
bottleneck K=16 mean   0.7235
```

So a large amount of local computation can live behind a much narrower interface in this toy; **the compression operator matters**.

### 2. A mixer after collapse is too late / too redundant

The post-collapse mixer does not rescue the branch representation and becomes catastrophically poor at K=16. This should not be generalized to all mixers. The specific lesson is architectural:

> If Y learns a receiver, the learned map must touch the rich local state **before** the irreversible narrow readout. Spending budget only after that readout cannot restore distinctions it discarded, and a plain linear post-map is largely redundant with the next layer anyway.

This is exactly the distinction that the receiver-relative work in `Dig` makes operational: changing computation downstream of a fixed readout is not the same thing as changing the readout itself.

### 3. No dendritic privilege has been earned

At K=4 the fixed branch model is modestly ahead of the ordinary bottleneck on this synthetic three-seed mean. At K=16 the ordinary bottleneck is dramatically stronger. There is no basis here for claiming a special dendritic mechanism.

## Hardware note

All narrow reference implementations are slower than the point baseline in this CPU experiment because they execute extra local operations and materialize intermediate state. **Logical receiver width is not hardware efficiency.**

The Wu et al. public implementation already contains a fused Triton branch-matmul -> ReLU -> branch-reduction kernel. Y should not claim a hardware win until a comparable fused/local implementation is profiled on the same workload.

## Decision

**Survives:** communication-bounded local computation.  
**Not earned:** fixed dendritic pooling as the privileged mechanism.  
**Killed:** “K=16 failed, therefore quarter-width is intrinsically too narrow.”  
**Banked null:** post-collapse learned mixing as a good use of the same budget.

The next architecture gate should explore the **pre-collapse reducer frontier**: how much of a fixed parameter/compute budget should create local nonlinear features, and how much should be spent on a learned sparse/dense receiver that sees those features before they are discarded?

Mandatory endpoints are the fixed branch reducer and the ordinary dense bottleneck. Do not tune the synthetic teacher to manufacture a branch win.
