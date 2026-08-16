# Gate 2 — cost/locality protocol

**Date:** 2026-08-16  
**Status:** capacity/fairness audit completed; hardware/locality gate open.

## Why Gate 2 exists

Y began by asking whether an interior learned pre-collapse receiver could beat
both a fixed dendritic-style reducer and an ordinary dense bottleneck under a
hard communication width.

Gate 1 did **not** find such an interior winner. Gate 2 then uncovered a more
important methodological fact: the historical fixed receiver used branch
**mean**, whereas the motivating Wu-style block is a branch **sum**. Those maps
have identical connectivity and differ only by a constant factor, but the
constant badly changed optimization in Y's unnormalized MLP.

The scale-controlled audit in `docs/GATE2_FAIRNESS_RECEIPT.md` therefore
supersedes the old strong interpretation of the fixed-branch null.

After parameter-free normalization, dense, grouped, and fixed local aggregation
all occupied essentially the same Digits accuracy band. No special learned
receiver was established.

So Gate 2 is now the systems question it should be:

> **If local nonlinear aggregation can preserve task capacity at a narrow
> receiver, can the wide local state actually remain local on real hardware?**

Y is still not allowed to add routing, growth, temporal state, or another
biological nonlinearity as a rescue.

---

## Paper boundary

Wu et al. motivate active-dendrite-style computation as localized nonlinear
aggregation: many branch/synaptic computations feed a narrower neuron output.
The engineering claim is not that a narrow logical tensor automatically means
less hardware traffic. The extra local synaptic access and aggregation have
costs too.

Y therefore keeps these quantities separate:

```text
logical receiver width
learned parameter count
multiply/add count
wide local activation width
whether that activation is materialized
PyTorch allocated memory
kernel count / launch behavior
wall-clock latency
physical DRAM traffic
energy
```

A reference PyTorch graph may expose only R values between modules while still
writing the full local H-vector to global memory. Such a graph has a narrow API,
not yet a communication advantage.

Physical DRAM traffic needs hardware counters / an appropriate profiler.
PyTorch peak allocation alone is not a traffic measurement.

---

# Matched hidden budget

For receiver width `R` and hidden budget factor `K`:

```text
B = K * R^2
```

All Gate-2 blocks expose exactly `R` values to the next block.

## Dense endpoint

```text
R -> H -> R
     ReLU

H = K*R/2
weights = R*H + H*R = B
```

This is the ordinary bottleneck control.

## Low-rank reducer control

```text
R -> H -> q -> R
     ReLU   linear factorization

weights = R*H + H*q + q*R <= B
```

There is no nonlinearity between the two reducer factors, so the receiver is
rank `q`. Gate 2 variance-matches the factor initialization to avoid an
artificial short-run disadvantage.

## Grouped/block reducer control

```text
R -> H -> R
     ReLU

H -> R is block diagonal with G groups
```

Cost:

```text
R*H + R*H/G <= B
```

`G=1` is exactly the dense endpoint. Increasing G gives each receiver group a
more local subset of H and spends the saved reducer weights on additional
private nonlinear features.

This is a standard grouped/block-linear control, not a Y novelty claim.

## Fixed local aggregation endpoint

Paper-form topology:

```text
R -> K*R -> grouped SUM -> R
     ReLU
```

All learned budget goes into local feature generation and the reducer has no
learned weights.

Historical Y used grouped **mean**. Keep mean only as a scale diagnostic.
For architecture/cost claims, mean and sum have the same connectivity; for
numerical training behavior they are not interchangeable in an unnormalized
deep stack.

---

# Phase A — capacity/fairness audit: COMPLETE

See `docs/GATE2_FAIRNESS_RECEIPT.md`.

The decisive scale-controlled three-seed / 15-epoch Digits means were:

```text
dense             0.961111
grouped2          0.962963
grouped4          0.973457
lowrank4          0.947531
lowrank8          0.954321
lowrank16         0.953086
fixed_mean        0.969753
fixed_sqrt_sum    0.974074
fixed_sum         0.973457
```

No candidate established a significant accuracy advantage at n=3. More
important, the enormous unnormalized mean-vs-sum gap disappeared after
parameter-free normalization.

Capacity-side verdict:

```text
special learned receiver needed          NO
fixed local aggregation ruled out        NO
interior grouped winner established       NO
hardware locality advantage established  NOT YET
```

Do not spend the next cycle extending Digits architecture sweeps unless the
hardware gate exposes a concrete reason to discriminate among these blocks.

---

# Phase B — actual CUDA cost gate

The hardware gate uses larger receiver sizes than Digits so launch overhead
does not completely dominate.

Two measurements are required.

## B1. Single-block microcost

Benchmark one receiver-to-receiver block repeatedly on the same input:

```text
R = 256 by default
K = 16
batch = 1, 8, 32, 128, 512
FP32 and FP16 separately
forward latency
forward+backward latency
peak allocated-memory delta
```

This permits the exact raw `fixed_sum` operator without repeatedly multiplying
activation scale through a deep stack.

Shortlist:

```text
dense
fixed_sum
grouped2
grouped4
lowrank16
```

## B2. Scale-controlled stack

Benchmark several hidden blocks with the same parameter-free normalization used
in the capacity audit:

```text
(block -> LayerNorm(no affine)) x depth
```

All candidates receive the same normalization overhead. This tests end-to-end
receiver flow without making raw branch-sum scale itself the benchmark.

Report eager and `torch.compile` separately because compiler fusion can change
the conclusion.

---

## What a useful hardware result looks like

### Outcome A — dense wins both eager and compiled

No efficient Y block yet. Preserve the null. Do not add routing.

### Outcome B — low rank spans the best frontier

An established factorization explains the useful tradeoff. That is a practical
result, not a new primitive.

### Outcome C — grouped receiver matches cost/accuracy best

Useful but still standard. Replicate on a second workload and compare against
optimized grouped-linear implementations before claiming anything new.

### Outcome D — fixed local topology preserves accuracy but eager PyTorch loses

This is a systems diagnosis, not an architecture failure. Inspect profiler
traces. If the wide `K*R` activation is materialized and re-read, the next valid
experiment is a fused projection + nonlinearity + local reduction kernel.

### Outcome E — fused/local fixed block is materially cheaper

That would replicate the motivating locality principle in Y's environment.
It still is not Y novelty. Replicate on another workload/size before reopening
learned/context-dependent receivers.

### Outcome F — an interior structured receiver is Pareto-better after honest
hardware controls

Only then reopen the question of whether the receiver itself should become
context dependent.

---

## Frozen prohibition

Until Phase B has a real measured result, do not add:

```text
WAIT / ROUTE / PROBE controller
receiver bank
structural growth
continual-learning plasticity
WidePresent state
new dendritic activation functions
```

The project currently lives or dies on an ordinary systems question:

> **Can useful computation stay rich locally while the boundary stays narrow
> without paying the same or greater cost somewhere else?**
