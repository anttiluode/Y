# Gate 2 — cost/locality protocol

**Date:** 2026-08-16  
**Status:** capacity/fairness audit completed; hardware/locality gate open.

## Why Gate 2 exists

Y began by asking whether an interior learned pre-collapse receiver could beat
both a fixed dendritic-style reducer and an ordinary dense bottleneck under a
hard communication width.

Gate 1 did **not** find such an interior winner. Gate 2 then uncovered a more
important methodological fact: historical Y used branch **mean**, whereas the
motivating Wu-style fixed local block is a branch **sum**. Those maps have the
same connectivity and differ only by a constant factor, but the factor badly
changed optimization in Y's unnormalized MLP.

After parameter-free normalization, dense, grouped, and fixed local aggregation
all occupied essentially the same Digits accuracy band. No special learned
receiver was established. See `docs/GATE2_FAIRNESS_RECEIPT.md`.

So Gate 2 is now the systems question it should be:

> **If local nonlinear aggregation can preserve task capacity at a narrow
> receiver, can the wide local state actually remain local on real hardware?**

Do not add routing, growth, temporal state, or another biological nonlinearity
to rescue this gate.

---

# Paper boundary and accounting

The exact GPU accounting is banked in `docs/GATE2_GPU_ACCOUNTING.md`.

For a point GEMM

```text
A: M x L
B: L x N
C: M x N
```

and K dendrites, the equal-compute local shape is approximately

```text
A_hat: M x L/sqrt(K)
B_hat: L/sqrt(K) x N*sqrt(K)
pre-collapse C_hat: M x N*sqrt(K)
post-collapse output: M x N/sqrt(K)
```

with K-way nonlinear local summation.

A crucial subtlety from the paper's GPU appendix is that, **before cache reuse
is exploited**, the equal-compute point and dendritic shapes have the same
first-order global-memory read count; the immediate saving is the narrower
result write. The larger read advantage arises from cache-aware block grouping
and therefore depends on working-set size / cache behavior.

That is why Y must sweep sizes rather than report one convenient latency number.

Keep these quantities separate:

```text
logical boundary width
learned parameter count
learned MAC count
wide local activation width
whether that activation is materialized
PyTorch active allocation
kernel count / launch behavior
wall-clock latency
physical DRAM reads/writes
L2 hit behavior
energy
```

A narrow module return value is not evidence of lower DRAM traffic.

---

# Matched learned budget

For narrow receiver width `R` and budget factor `K`:

```text
B = K * R^2
```

## Paper-equivalent wide point control

Let

```text
D = R * sqrt(K)
```

Then

```text
wide_point: D -> D
weights = D^2 = K*R^2 = B
```

This is intentionally wider at the block boundary. It exists only for the
paper-replication axis.

## Ordinary narrow dense bottleneck

```text
R -> H -> R
     ReLU
H = K*R/2
weights = R*H + H*R = B
```

This is Y's strongest boring practical control: it already exposes only R
values.

## Fixed local aggregation

```text
R -> K*R -> grouped SUM -> R
     ReLU
weights = K*R^2 = B
```

The reducer is deterministic. Historical `mean` remains only as a scale audit;
`sum` is the paper-form topology used for the hardware axis.

## Grouped/block learned receiver

```text
R -> H -> R
     ReLU
H -> R is block diagonal with G groups
R*H + R*H/G <= B
```

`G=1` is the dense endpoint. Increasing G makes the learned receiver more local
and spends saved reducer weights on more private nonlinear features.

## Low-rank receiver

```text
R -> H -> q -> R
     ReLU   linear factorization
R*H + H*q + q*R <= B
```

This is a standard factorized control, not a Y novelty claim.

---

# Phase A — capacity/fairness audit: COMPLETE

Scale-controlled three-seed / 15-epoch Digits means:

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

No candidate established a credible accuracy advantage at n=3. The important
result is that the dramatic unnormalized mean-vs-sum difference disappeared
once receiver scale was controlled.

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

# Phase B — actual CUDA gate

Canonical instrument:

```text
experiments/gate2_hardware_shortlist.py
```

It has **two distinct comparison axes**.

## Axis A — paper replication

```text
wide_point  vs  fixed_sum
```

Matched:

```text
learned parameters = K*R^2
learned GEMM MACs per sample = K*R^2
```

Different by design:

```text
wide_point boundary = R*sqrt(K)
fixed_sum boundary  = R
```

For K=16, `fixed_sum` exposes one quarter as many boundary activations.

If this axis wins, Y has replicated the motivating communication geometry. That
is useful, but it is **not Y novelty**.

## Axis B — Y's harder practical control

```text
dense narrow bottleneck  vs  fixed_sum
```

Both already expose the same R-wide boundary and both use the same learned
budget.

This asks the harder question:

> **When an ordinary bottleneck already communicates only R values, is
> deterministic local collapse actually cheaper to realize?**

If dense is as fast / memory-friendly, dendritic language bought Y no useful
block even if Axis A reproduces the paper.

Grouped2, grouped4, and lowrank16 remain standard controls around Axis B.

---

# Size/cache sweep is mandatory

Default reference widths:

```text
R = 256, 512, 1024
K = 16
```

so the paper-axis wide point widths are:

```text
D = 1024, 2048, 4096
```

Default batches:

```text
32, 128, 512
```

The size sweep is part of the hypothesis. Small/cache-resident workloads may
show little advantage; the paper's interesting global-memory behavior appears
when working sets cross cache regimes.

Reference forward run:

```bash
python experiments/gate2_hardware_shortlist.py
```

Add training/backward cost:

```bash
python experiments/gate2_hardware_shortlist.py --backward
```

FP16 separately:

```bash
python experiments/gate2_hardware_shortlist.py --dtype float16
```

Scale-controlled stacks:

```bash
python experiments/gate2_hardware_shortlist.py --modes stack --receiver-widths 256 512
```

Compiler experiment:

```bash
python experiments/gate2_hardware_shortlist.py --compile
```

Keep eager and compiled tables separate.

---

# Fusion is the real Y systems hypothesis

Naive eager fixed aggregation may legitimately lose.

Dense bottleneck temporary state:

```text
H = K*R/2
```

Fixed local temporary state if materialized:

```text
K*R
```

At the same learned MAC budget the naive fixed temporary is twice as wide as
the dense bottleneck temporary. Therefore a PyTorch graph that does

```text
Linear -> ReLU -> reshape -> sum
```

and writes the whole `K*R` tensor to global memory has **not** implemented the
locality hypothesis; it has merely hidden a wide tensor inside a narrow module.

The topology becomes interesting only if an implementation can approximate

```text
GEMM tile
 -> activation
 -> deterministic K-way local reduction
 -> write only R values
```

without globally materializing the branch state.

A custom fused fixed-local kernel is allowed only if the reference profiling
identifies materialization/re-read as the bottleneck worth removing.

And a fused fixed-local kernel must be compared against an **optimized/fused
ordinary MLP/bottleneck**, not only eager PyTorch. If the optimized MLP matches
or beats it, Y's primitive is occupied / killed.

---

# Verdict rules

```text
A. wide_point loses to fixed_sum, but dense <= fixed_sum
   -> paper principle replicated; no Y block.

B. low-rank/grouped spans the best practical frontier
   -> established method explains result; no Y novelty.

C. fixed_sum loses in eager and profiler shows K*R materialization dominates
   -> fused local-collapse replication is allowed.

D. fixed_sum loses after an appropriate local implementation
   -> stop the architecture branch; do not rescue it with routing.

E. fused fixed_sum beats an optimized/fused dense bottleneck at matched
   narrow boundary + learned work
   -> genuine Y systems candidate; replicate another workload/size before
      reopening context-dependent receivers.
```

Physical DRAM-byte, L2-hit, or energy claims require hardware counters / an
appropriate profiler; PyTorch allocation peaks alone are insufficient.

---

## Frozen prohibition

Until Phase B resolves, do not add:

```text
WAIT / ROUTE / PROBE controller
receiver bank
structural growth
continual-learning plasticity
WidePresent state
new dendritic activation functions
```

The project currently lives or dies on the ordinary systems question:

> **Can useful computation stay rich locally while the boundary stays narrow
> without paying the same or greater cost somewhere else?**
