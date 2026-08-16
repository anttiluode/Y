# Gate 2 — cost/locality protocol

**Date:** 2026-08-16  
**Status:** instrument under construction; no Gate-2 result yet.

## Why Gate 2 exists

Gate 1 killed the first candidate Y block on the first external dataset.  At a
quarter-width receiver, cheap sparse pre-collapse correction did not preserve
the ordinary dense bottleneck's Digits accuracy.  The 50% receiver-budget
candidate merely tied the dense bottleneck.

So Y is not allowed to add routing, growth, temporal state, or another
biological nonlinearity as a rescue.

The surviving question is more ordinary:

> **If a rich learned pre-collapse reducer is required, can standard
> structured/local reducers preserve the dense bottleneck's accuracy while
> materially changing measured hardware cost?**

This gate is deliberately a control gate.  Low rank, grouped/block linear
layers, and fixed branch aggregation are prior art / standard mechanisms.  A
win by one of them is useful even if it leaves Y with no novel primitive.

---

## Paper boundary

Wu et al. (2026), *Dendritic nonlinearities mitigate communication costs*,
argue that the useful advantage of active-dendrite-style architectures is not
extra capacity once model complexity is controlled, but localized nonlinear
aggregation that can reduce the width communicated onward.  Their analysis
also makes clear that the extra local synaptic access has a cost of its own;
shrinking the inter-layer vector is not automatically equivalent to lower
physical memory traffic.

Y therefore keeps these quantities separate:

```text
logical receiver width
learned parameter count
multiply/add count
local activation width
PyTorch allocated memory
kernel count / launch behavior
wall-clock latency
physical DRAM traffic
energy
```

Only the middle hardware quantities are measured by the first Gate-2 harness.
Physical DRAM traffic needs Nsight/CUPTI-class counters or an equivalent
hardware profiler.

---

## Matched hidden budget

For receiver width `R` and hidden budget factor `K`:

```text
B = K * R^2
```

All Gate-2 blocks expose exactly `R` values to the next block.

### Dense endpoint

```text
R -> H -> R
H = K*R/2
weights = R*H + H*R = B
```

This is the ordinary bottleneck that won Gate 1.

### Low-rank reducer control

```text
R -> H -> q -> R
       ReLU   linear factorization

weights = R*H + H*q + q*R <= B
```

`H` is chosen as large as possible under the budget.  There is no nonlinearity
between the two reducer factors, so the receiver really is rank `q`.

### Grouped/block reducer control

```text
R -> H -> R
       ReLU

H -> R reduction is block diagonal with G groups.
```

Its learned cost is

```text
R*H + R*H/G <= B.
```

`G=1` is exactly the dense endpoint.  Increasing `G` makes the learned receiver
more local and spends the saved reducer weights on more private nonlinear
features.  This gives a clean standard-control frontier rather than inventing
another bespoke sparse receiver.

### Fixed branch endpoint

```text
R -> K*R -> grouped mean -> R
       ReLU
```

All learned budget goes into local feature generation; the reducer is fixed.
This is the Wu-style endpoint and the Gate-0/Gate-1 null control.

---

# Phase A — accuracy gate

Dataset: scikit-learn Digits, matching Gate 1.

Default decisive settings:

```text
R = 32
K = 16
receiver ratio vs D=128 = 0.25x
hidden depth = 4
10 stratified split seeds
40 epochs
paired minibatch order across architectures
```

Sweep:

```text
dense
lowrank q = 4, 8, 16
grouped G = 2, 4, 8, 16, 32
fixed branch
```

Run:

```bash
python experiments/gate2_cost_locality.py --accuracy-only
```

Quick smoke:

```bash
python experiments/gate2_cost_locality.py --accuracy-only --quick
```

The accuracy result must be read together with each block's actual parameter
count and budget slack.  Small integer rounding differences are reported, not
hidden.

---

# Phase B — actual CUDA cost gate

Digits is too small to be a trustworthy GPU microbenchmark.  The hardware gate
therefore benchmarks receiver-to-receiver cores at larger widths and batch
sizes while keeping the same architectural budget rule.

Default microbenchmark:

```text
R = 256
K = 16
8 repeated hidden blocks
batch = 1, 8, 32, 128, 512
CUDA events for synchronized timing
forward latency
optional forward+backward latency
peak allocated-memory delta
```

Run on the local NVIDIA GPU:

```bash
python experiments/gate2_cost_locality.py --hardware-only --bench-backward
```

Half precision is a separate measurement, not mixed into the float32 table:

```bash
python experiments/gate2_cost_locality.py --hardware-only --bench-backward --bench-dtype float16
```

Optional compiler comparison:

```bash
python experiments/gate2_cost_locality.py --hardware-only --compile
```

Compiler results must be reported separately from eager results because fusion
can change the conclusion.

Optional PyTorch kernel/allocation trace for one model:

```bash
python experiments/gate2_cost_locality.py \
  --hardware-only \
  --profile-model grouped4 \
  --profile-batch 128 \
  --trace artifacts/gate2_grouped4_trace.json
```

The trace is for kernel structure and PyTorch memory behavior.  It is **not** a
physical DRAM-byte measurement.

---

## Gate-2 verdict rules

### Outcome A — dense wins accuracy and hardware

Stop.  The current Y architecture search has no efficient block.  Keep the
receiver framing and negative results; do not invent routing to rescue it.

### Outcome B — low rank spans the best frontier

Also mostly stop.  That says an established low-rank receiver explains the
useful tradeoff.  Y may retain a benchmarking/framing contribution, but not a
new primitive.

### Outcome C — grouped/block receiver matches dense accuracy and is cheaper

Interesting, but still not a novelty claim.  First replicate on a larger task,
then compare optimized/fused implementations.  Only after standard grouped
linear baselines are exhausted should Y ask whether context-dependent receiver
selection adds anything.

### Outcome D — fixed branch becomes cheaper only after fusion

That is primarily a replication of Wu et al.'s locality claim.  Useful, but not
Y novelty.  The next question would be whether a learned local reducer can be
fused without giving back the communication advantage.

### Outcome E — an interior structured receiver has a robust Pareto win

This is the only outcome that re-opens Gate 3.  The candidate must preserve
accuracy while improving a measured hardware cost on more than one workload.
Then, and only then, test whether the receiver should become context dependent.

---

## Frozen prohibition

Until Gate 2 has a real result, do not add:

```text
WAIT / ROUTE / PROBE controller
receiver bank
structural growth
continual-learning plasticity
WidePresent state
new dendritic activation functions
```

The project currently lives or dies on an ordinary systems question:

> **Can useful computation stay rich locally while the learned boundary stays
> narrow without paying the same or greater cost somewhere else?**
