# Gate 0 — first receipt

**Date:** 2026-08-16  
**Status:** partial survival at 2x receiver compression; 4x receiver compression fails this first toy.

This is a synthetic instrument check, not an external benchmark and not a hardware-efficiency result.

## Question

At approximately matched parameter count, can local nonlinear branch computation preserve held-out classification performance while reducing the width communicated between hidden blocks?

The comparison uses a point MLP of width `D=128` and branch-reduce MLPs with square hidden-core complexity matched by

```text
K * R^2 ~= D^2
```

so:

```text
K=4   -> R=64   -> 0.5x logical receiver traffic
K=16  -> R=32   -> 0.25x logical receiver traffic
```

For the fixed input boundary, the branch models use `sqrt(K)` branches. That keeps the first-layer weight count matched to the point model rather than accidentally over-allocating it.

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

The run used the development container with `torch 2.10.0+cpu`; therefore timing here is only reference-PyTorch timing, not a GPU traffic measurement.

## Results

| seed | point acc | K=4 acc | K=16 acc |
|---:|---:|---:|---:|
| 0 | 0.7983 | 0.7954 | 0.7002 |
| 1 | 0.7402 | 0.7236 | 0.6631 |
| 2 | 0.7495 | 0.7188 | 0.6143 |
| **mean** | **0.7627** | **0.7459** | **0.6592** |

Parameter counts:

```text
point       54,272
K=4         53,760   (0.991x)
K=16        53,504   (0.986x)
```

Hidden square-block weights are exactly `16,384` in all three cases.

Mean accuracy deltas versus point:

```text
K=4     -0.0167 absolute accuracy   at 0.50x receiver width
K=16    -0.1035 absolute accuracy   at 0.25x receiver width
```

## What this says

The first toy does **not** show a free 4x interface compression. `K=16` loses too much accuracy and is provisionally a failure here.

`K=4` is more interesting: roughly matched total parameters and half the communicated hidden width cost about 1.7 percentage points on the three-seed mean. That is close enough to justify a real-data replication, but not close enough to call equivalent performance.

The naive PyTorch branch implementation is also slower than the point baseline in this CPU run because it materializes the expanded local branch state. That is expected and important: **logical receiver width is not hardware efficiency.** The Wu et al. reference implementation includes a fused Triton branch-matmul/nonlinearity/reduction kernel specifically to keep the local expansion from becoming expensive global traffic.

## Decision

**Survives narrowly:** carry `K=4` to an external-data / stronger-baseline gate.  
**Fails for now:** do not tune this synthetic teacher to rescue `K=16`.

Before any receiver-routing invention, the next engineering gate should answer two boring questions:

1. Can we reproduce the fixed `K=4` tradeoff on a standard dataset with matched training?
2. On actual GPU hardware, does a fused/local implementation turn the narrower receiver into measured memory-traffic or latency savings?

Only after those pass should Y spend complexity on dynamic receiver choice.
