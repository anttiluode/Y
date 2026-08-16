# Gate 2 — GPU accounting target

**Date:** 2026-08-16  
**Purpose:** specify what the hardware experiment is actually trying to measure before any custom kernel is written.

## 1. What Wu et al. actually count

The GPU appendix of Wu et al., arXiv:2306.11950v2 / Patterns 2026, analyzes a
standard feed-forward GEMM

```text
C = sigma(A B)
A: M x L
B: L x N
C: M x N
```

against an equal-compute dendritic-shaped GEMM. For K dendrites, the equivalent
layer shrinks the communicated input/output channel dimensions by `sqrt(K)` and
expands the pre-aggregation output so that learned matrix-multiply complexity
is unchanged:

```text
A_hat: M x (L/sqrt(K))
B_hat: (L/sqrt(K)) x (N*sqrt(K))
C_hat before aggregation: M x (N*sqrt(K))
C_hat after grouping K outputs: M x (N/sqrt(K))
```

The post-activation branch outputs are **summed in groups of K**.

This mapping is exactly why Y's paper-axis controls are:

```text
wide_point: D -> D
fixed_sum : R -> K*R -> grouped sum -> R

D = R*sqrt(K)
D^2 = K*R^2
```

They have the same learned-weight / GEMM-MAC budget.

---

## 2. The paper's first-order GPU result is subtler than 'narrow = cheap'

Ignoring L2 reuse and holding GEMM block sizes fixed, their derivation gives:

```text
point global-memory read cost      == dendritic read cost
point result write                 = M*N
dendritic result write             = M*N/sqrt(K)
```

So the naive equal-compute shape change does **not** automatically reduce the
dominant matrix reads. It immediately reduces the post-aggregation write, but
that alone may be a minor gain.

The larger predicted read advantage comes from cache-aware block grouping.
Because the dendritic-shaped GEMM has a shorter reduction dimension
`L/sqrt(K)`, a block group can reuse operands differently within a fixed L2
capacity `Q`. Their simplified cache model predicts approximately

```text
dendritic global-memory reads ~= point reads / sqrt(K)
```

when the grouped working set is in the regime where the cache argument applies.

That makes **working-set size** a first-class experimental variable.

---

## 3. Their empirical target

The paper's GPU appendix reports an Nvidia A40 experiment using Triton-style
block GEMM and Nvidia Nsight Compute global-memory measurements. It searches
block dimensions and a grouping parameter rather than assuming one tile shape.

Their qualitative result is size dependent:

```text
small matrices / cache-resident     weaker-than-theory advantage
working set beyond L2               behavior approaches ~1/sqrt(K) read scaling
very large matrices                 cache eviction can worsen the ideal prediction
```

They also explicitly report that lower communication does not guarantee lower
runtime: at small problem sizes, large K can be slower because extra nonlinear
activations run outside the Tensor Core matrix multiply path.

Therefore a single `R=256`, single-batch latency number cannot test the paper's
claim.

---

# 4. Two different questions Y must measure

## Axis A — paper replication

```text
wide_point  vs  fixed_sum
```

Matched:

```text
learned weights
learned GEMM MACs per sample
```

Different by design:

```text
wide_point boundary = R*sqrt(K)
fixed_sum boundary  = R
```

For K=16, the logical communicated feature width is 4x smaller in `fixed_sum`.
This axis asks whether the paper's matrix-shape/local-aggregation principle is
visible in Y's environment.

A positive result here is a **replication**, not Y novelty.

## Axis B — Y's harder practical control

```text
dense narrow bottleneck  vs  fixed_sum
```

Matched:

```text
boundary width = R
learned-weight budget = K*R^2
learned MAC budget approximately the same
```

The dense block is

```text
R -> H -> R
H = K*R/2
ReLU between the two learned GEMMs
```

The fixed block is

```text
R -> K*R -> grouped SUM -> R
ReLU before deterministic reduction
```

This asks a stronger question than the paper:

> **If an ordinary architecture already gives us the same narrow interface,
> is deterministic local collapse cheaper to realize than a learned dense
> bottleneck?**

If the answer is no, dendritic language buys Y no useful block even if the
paper replication succeeds.

---

# 5. Why fusion is the actual Y systems hypothesis

Consider only the hidden activation traffic in a naive unfused implementation.

Dense bottleneck:

```text
GEMM1 produces H = K*R/2 activations
ReLU(H)
GEMM2 consumes H
```

If `H` is globally materialized, it is written and read between GEMMs.

Fixed local block:

```text
one GEMM produces K*R branch activations
ReLU(K*R)
grouped reduction consumes them
```

If the branch tensor is globally materialized, fixed can be **worse** because
its temporary state is twice the dense bottleneck's H at the same learned MAC
budget.

Therefore Y gets no credit for a narrow return tensor if PyTorch writes the
whole `K*R` branch tensor to global memory first.

The useful implementation would perform something like

```text
GEMM tile
 -> activation
 -> deterministic K-way local reduction
 -> write only R-wide result
```

with branch values retained in registers/shared memory or otherwise reduced
before expensive global materialization.

This is the concrete meaning of

> compute locally; communicate the collapse.

---

# 6. Mandatory control if Y writes a fused kernel

A fused fixed-local kernel cannot be compared only against eager
`Linear -> ReLU -> Linear` PyTorch and called a win.

The dense bottleneck also has a strong optimization opportunity: fused MLP
implementations can tile/fuse portions of the activation and second projection,
reducing intermediate traffic.

So if a custom fixed kernel survives the reference benchmark, the decisive
systems comparison becomes:

```text
optimized/fused fixed local collapse
vs
optimized/fused ordinary dense MLP/bottleneck
```

at matched learned work and boundary width.

If the ordinary fused MLP matches or beats it, Y's block is occupied / killed.

---

# 7. Current instrument

`experiments/gate2_hardware_shortlist.py`

Default candidates:

```text
wide_point     paper point baseline
dense          ordinary narrow bottleneck
fixed_sum      paper-form fixed local aggregation
grouped2       standard mild block-local receiver
grouped4       standard stronger block-local receiver
lowrank16      standard factorized receiver
```

Default reference widths:

```text
R = 256, 512, 1024
```

with K=16, so the paper-axis wide point widths are:

```text
D = 1024, 2048, 4096
```

The size sweep is intentional: it looks for the cache/working-set crossover
rather than averaging it away.

### Micro forward

```bash
python experiments/gate2_hardware_shortlist.py
```

### Include backward

```bash
python experiments/gate2_hardware_shortlist.py --backward
```

### FP16 separately

```bash
python experiments/gate2_hardware_shortlist.py --dtype float16
```

### Scale-controlled stacks

```bash
python experiments/gate2_hardware_shortlist.py --modes stack --receiver-widths 256 512
```

### Compiler experiment

```bash
python experiments/gate2_hardware_shortlist.py --compile
```

Compiler results must remain separate from eager results.

---

# 8. What the reference instrument can and cannot prove

It can measure:

```text
wall-clock CUDA event time
PyTorch active-allocation peaks
working-set / size crossovers
whether a stock compiler changes the relative ordering
```

It cannot by itself prove:

```text
physical DRAM read/write bytes
L2 hit rates
shared-memory traffic
register pressure
energy
```

Those require Nvidia Nsight Compute / CUPTI-class counters or an equivalent
hardware profiler.

The next custom-kernel step is permitted **only if** the reference profiler
shows that materializing / rereading the branch state is the bottleneck worth
removing.

No routing, structural growth, or context-dependent receiver is justified by
this gate.
