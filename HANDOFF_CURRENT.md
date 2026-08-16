# Y — CURRENT HANDOFF

**Updated:** 2026-08-16  
**Branch:** `agent/communication-bounded-blocks`  
**PR:** draft #1  
**Status:** capacity-side receiver question corrected; GPU locality gate is now the active experiment.

## Read first

1. `docs/GATE2_FAIRNESS_RECEIPT.md` — the critical correction: old fixed-mean failures were scale-confounded.
2. `docs/GATE2_GPU_ACCOUNTING.md` — **current next-step specification**: what Wu et al. actually count on GPU and the two hardware axes Y must test.
3. `experiments/gate2_hardware_shortlist.py` — CUDA instrument with equal-budget paper baseline + Y practical controls.
4. `docs/GATE2_COST_LOCALITY_PROTOCOL.md` — stop conditions.
5. `docs/GATE1_PRECOLLAPSE_FRONTIER_RECEIPT.md` — historical sparse-correction null, now deliberately narrowed in scope.
6. `docs/GATE0_FIRST_RECEIPT.md` — paired synthetic receipt; fixed-mean capacity interpretation superseded.

---

# One-line state

> **Y did not discover a special learned receiver. It discovered that its earlier fixed-branch null was mostly an optimization-scale artifact. Once scale is controlled, fixed local nonlinear aggregation, grouped receivers, and the ordinary dense bottleneck occupy the same small-task accuracy band. The serious question is now whether the local collapse can actually save data movement on hardware — and whether it can beat an ordinary narrow bottleneck, not merely a wide point layer.**

---

# 1. Critical correction to Gate 0 / Gate 1

Historical Y fixed branches used

```text
mean_k ReLU(branch_k)
```

The Wu-style block uses

```text
sum_k ReLU(branch_k)
```

For fixed K these have identical connectivity and differ only by a positive
constant scale. In an unnormalized deep MLP that constant radically changed
training.

Gate 2 fairness audit, 3 paired Digits splits / 15 epochs / R=32 / K=16:

```text
UNNORMALIZED

dense             0.921605 +/- 0.036774
grouped2          0.943827 +/- 0.007484
grouped4          0.922222 +/- 0.015822
lowrank4          0.910494 +/- 0.031878
lowrank8          0.924691 +/- 0.009321
lowrank16         0.935802 +/- 0.010851
fixed_mean        0.409877 +/- 0.292624
fixed_sqrt_sum    0.922222 +/- 0.005556
fixed_sum         0.961728 +/- 0.005953
```

The same fixed architecture moved from catastrophic to strong when only its
constant aggregation scale changed. Therefore this old sentence is withdrawn
as a capacity conclusion:

```text
fixed dendritic pooling loses to an ordinary learned bottleneck
```

The historically accurate statement is only:

```text
fixed-mean aggregation optimized badly in the old unnormalized Gate 0/1 setup
```

---

# 2. Scale-controlled capacity result

Gate 2 inserted parameter-free
`LayerNorm(R, elementwise_affine=False)` after each hidden receiver.

Three paired Digits splits / 15 epochs:

```text
dense             0.961111 +/- 0.009799
grouped2          0.962963 +/- 0.022453
grouped4          0.973457 +/- 0.008553
lowrank4          0.947531 +/- 0.011906
lowrank8          0.954321 +/- 0.012330
lowrank16         0.953086 +/- 0.019275
fixed_mean        0.969753 +/- 0.014384
fixed_sqrt_sum    0.974074 +/- 0.009259
fixed_sum         0.973457 +/- 0.009135
```

At n=3 no candidate established a credible accuracy advantage. Among the three
identical-connectivity fixed variants, normalization collapsed the scale gap:

```text
fixed_sqrt_sum vs fixed_mean   +0.004321   p=0.779848
fixed_sum      vs fixed_mean   +0.003704   p=0.800000
```

Current capacity verdict:

```text
special learned reducer required       NO
fixed local aggregation ruled out      NO
interior grouped winner established    NO
dense bottleneck uniquely superior     NO
```

This is the useful convergence with Wu et al.: the interesting value of local
nonlinear aggregation is not a magical capacity gain. It has to earn its keep
through locality / communication.

---

# 3. Gate 1 remains banked, but only for the exact tested object

Gate 1 tested:

```text
wide local ReLU state
-> fixed grouped-MEAN base receiver
-> zero-initialized sparse learned correction
```

The correction was zero-initialized, while the dense bottleneck reducer was
normally/randomly initialized.

Therefore Gate 1 cleanly says:

> **A cheap zero-initialized sparse correction did not grow out of the
> historical fixed-mean receiver and match the dense bottleneck until the
> reducer budget reached the dense endpoint.**

It does **not** prove that every sparse/local learned receiver is intrinsically
worse than dense reduction.

Do not reopen Gate 1 now. Hardware comes first.

---

# 4. What the paper's GPU argument actually says

This was re-read directly from the GPU appendix before building the next gate.

For a point GEMM

```text
A: M x L
B: L x N
C: M x N
```

and K dendrites, the equal-compute local model changes the shape to approximately

```text
A_hat: M x L/sqrt(K)
B_hat: L/sqrt(K) x N*sqrt(K)
pre-collapse C_hat: M x N*sqrt(K)
post-collapse output: M x N/sqrt(K)
```

with K-way nonlinear local summation.

Important subtlety:

```text
WITHOUT L2 REUSE OPTIMIZATION
point global-memory reads      == dendritic reads
point output write              = M*N
dendritic output write          = M*N/sqrt(K)
```

The larger predicted read win appears only when block processing exploits L2
reuse. The paper's simplified cache argument predicts roughly `1/sqrt(K)`
global-memory reads in the favorable regime.

Their empirical A40 study therefore searches tile/group settings and measures
global memory with Nsight Compute. Small/cache-resident matrices show weaker
gains; matrices beyond L2 approach the predicted scaling; very large working
sets deviate again due eviction. Runtime gains also appear mainly when memory
I/O is the bottleneck.

**Therefore size/cache crossover is part of the hypothesis, not noise.**

---

# 5. The two hardware axes Y must not confuse

## Axis A — paper replication

```text
wide_point  vs  fixed_sum
```

Let the fixed/local receiver be R wide. Then

```text
D = R*sqrt(K)
wide_point: D -> D
fixed_sum : R -> K*R -> nonlinear grouped sum -> R
```

Both use exactly

```text
K*R^2 learned weights / learned MACs per sample
```

For K=16 the fixed layer exposes one quarter as many boundary activations.

A win here is a **replication of the paper principle**, not Y novelty.

## Axis B — Y's harder practical question

```text
dense narrow bottleneck  vs  fixed_sum
```

Both already expose the same R-wide boundary and use the same learned budget.

Dense:

```text
R -> H -> R
H = K*R/2
ReLU between learned GEMMs
```

Fixed:

```text
R -> K*R -> grouped sum -> R
ReLU before deterministic reduction
```

This asks:

> **When a boring MLP bottleneck already communicates only R values, is the
> deterministic local collapse cheaper to realize?**

If not, Y has no useful block even if Axis A replicates Wu et al.

---

# 6. Fusion is the real systems hypothesis

Naive eager implementations can make fixed local aggregation look terrible for
a legitimate reason.

Dense bottleneck materializes roughly

```text
H = K*R/2
```

between two learned GEMMs.

Fixed local aggregation materializes

```text
K*R
```

branch values if implemented as ordinary `Linear -> ReLU -> reshape -> sum`.

At the same learned MAC budget the naive fixed temporary is twice as wide as the
dense bottleneck temporary. A narrow Python return tensor is therefore **not**
a hardware win.

The fixed topology becomes interesting only if the implementation can do
something like

```text
GEMM tile
 -> nonlinear activation
 -> deterministic K-way reduction
 -> write only R values
```

without writing the K*R branch tensor to global memory.

If Y eventually writes such a fused/local kernel, it must be compared against
an **optimized/fused ordinary MLP/bottleneck**, not only eager PyTorch. If the
fused MLP matches or beats it, Y's primitive is killed/occupied.

---

# 7. Current CUDA instrument

`experiments/gate2_hardware_shortlist.py`

Default candidates:

```text
wide_point     paper equal-compute point baseline
dense          ordinary narrow bottleneck
fixed_sum      paper-form fixed local aggregation
grouped2       standard block-local learned receiver
grouped4       stronger grouped control
lowrank16      standard factorized control
```

Default reference widths:

```text
R = 256, 512, 1024
K = 16
```

so the paper-axis point widths are

```text
D = 1024, 2048, 4096
```

Default batches:

```text
32, 128, 512
```

The size sweep is intentional.

### Reference forward microbenchmark

```bash
python experiments/gate2_hardware_shortlist.py
```

### Include training/backward cost

```bash
python experiments/gate2_hardware_shortlist.py --backward
```

### FP16 separately

```bash
python experiments/gate2_hardware_shortlist.py --dtype float16
```

### Scale-controlled stacked blocks

```bash
python experiments/gate2_hardware_shortlist.py --modes stack --receiver-widths 256 512
```

### Compiler comparison

```bash
python experiments/gate2_hardware_shortlist.py --compile
```

Keep eager and compiled tables separate.

The script records CUDA-event latency, PyTorch active-allocation peaks, actual
boundary widths and logical local widths. It deliberately does **not** infer
physical DRAM bytes or energy.

---

# 8. Next decision after the CUDA run

```text
A. wide_point loses to fixed_sum, but dense <= fixed_sum
   -> paper principle replicated; no Y block.

B. lowrank/grouped spans the best practical frontier
   -> standard mechanism explains result; no Y novelty.

C. fixed_sum loses in eager and profiler shows K*R materialization dominates
   -> fused local-collapse replication is allowed.

D. fixed_sum loses even after appropriate local implementation
   -> stop architecture branch; no routing rescue.

E. fused fixed_sum beats an optimized/fused dense bottleneck at matched
   narrow boundary + learned work
   -> genuine Y systems candidate; replicate another size/workload before
      reopening context-dependent receivers.
```

No routing, receiver bank, structural growth, WidePresent state, or new neuron
activation before this gate resolves.

No hardware-efficiency claim has been earned yet.
