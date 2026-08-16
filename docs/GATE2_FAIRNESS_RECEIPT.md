# Gate 2 fairness receipt — scale, initialization, and the fixed receiver

**Date:** 2026-08-16  
**Branch:** `agent/communication-bounded-blocks`  
**CI head:** `c26c551bbdeb18b76dfbd985cbc69b745240cc63`  
**CI run:** 31938320318  
**Status:** accuracy-side correction banked; hardware gate remains open.

## Why this receipt exists

Gate 0 and Gate 1 used `mean` aggregation in the fixed branch receiver. The
Wu-style dendritic block that motivated Y is a **sum** of branch nonlinearities.
Those two maps have identical connectivity and differ only by a positive
constant scale, but in an unnormalized deep MLP that scale can radically change
optimization.

Gate 1 also initialized its sparse learned correction at zero, while the dense
bottleneck reducer was randomly initialized. That is a valid test of whether a
cheap correction can *grow out of* the historical fixed receiver, but it is not
a clean best-possible sparse-reducer versus dense-reducer architecture test.

Before promoting the early `grouped2` bump or declaring fixed aggregation dead,
Gate 2 therefore audited:

- variance-matched low-rank factor initialization;
- fixed `mean`, `sum/sqrt(K)`, and raw `sum` aggregation;
- a parameter-free LayerNorm control after every hidden receiver to remove
  positive constant scale as an optimization confound.

The architecture and parameter budget of the three fixed variants are
identical. Any difference among them without normalization is a scale / training
dynamics result, not a representational-capacity result.

---

## Instrument

`experiments/gate2_fairness_audit.py`

Common setting:

```text
scikit-learn Digits
R = 32
K = 16
depth = 4
3 stratified split seeds: 0, 1, 2
15 epochs
AdamW lr = 2e-3
explicit paired minibatch order
```

Unnormalized command:

```bash
python experiments/gate2_fairness_audit.py
```

Scale-controlled command:

```bash
python experiments/gate2_fairness_audit.py --normalize-hidden
```

The normalized version inserts
`LayerNorm(R, elementwise_affine=False)` after each hidden block. It adds no
learned parameters and makes positive constant rescaling of a receiver largely
irrelevant (apart from epsilon / numerical effects).

---

# Result A — unnormalized

Mean test accuracy over three paired splits:

| model | mean accuracy | SD | delta vs dense | paired p |
|---|---:|---:|---:|---:|
| dense | 0.921605 | 0.036774 | — | — |
| grouped2 | 0.943827 | 0.007484 | +0.022222 | 0.334018 |
| grouped4 | 0.922222 | 0.015822 | +0.000617 | 0.969780 |
| lowrank4 | 0.910494 | 0.031878 | -0.011111 | 0.777421 |
| lowrank8 | 0.924691 | 0.009321 | +0.003086 | 0.865551 |
| lowrank16 | 0.935802 | 0.010851 | +0.014198 | 0.492633 |
| fixed_mean | 0.409877 | 0.292624 | -0.511728 | 0.100448 |
| fixed_sqrt_sum | 0.922222 | 0.005556 | +0.000617 | 0.979207 |
| fixed_sum | **0.961728** | 0.005953 | +0.040123 | 0.179316 |

Per-seed fixed variants:

| seed | mean | sum/sqrt(K) | sum |
|---:|---:|---:|---:|
| 0 | 0.100000 | 0.927778 | 0.968519 |
| 1 | 0.448148 | 0.922222 | 0.959259 |
| 2 | 0.681481 | 0.916667 | 0.957407 |

Scale-only differences versus `fixed_mean`:

```text
fixed_sqrt_sum   +0.512346   p=0.0967701
fixed_sum        +0.551852   p=0.0851863
```

### Interpretation

The historical fixed-mean failure is **not a robust capacity null**. The same
connectivity and same learned weights move from near chance to strong accuracy
when only the deterministic aggregation scale changes.

Raw `sum` looking best here is also **not a dendritic capacity win**. Since
`mean` and `sum` differ only by a constant factor, the enormous gap diagnoses
optimization / scale sensitivity in this unnormalized toy.

The early one-seed `grouped2 > dense` bump survives directionally at 15 epochs,
but with three seeds it is small and non-significant. It is not a Y block.

---

# Result B — parameter-free normalization control

With LayerNorm after every hidden receiver:

| model | mean accuracy | SD | delta vs dense | paired p |
|---|---:|---:|---:|---:|
| dense | 0.961111 | 0.009799 | — | — |
| grouped2 | 0.962963 | 0.022453 | +0.001852 | 0.894395 |
| grouped4 | 0.973457 | 0.008553 | +0.012346 | 0.289331 |
| lowrank4 | 0.947531 | 0.011906 | -0.013580 | 0.334250 |
| lowrank8 | 0.954321 | 0.012330 | -0.006790 | 0.623808 |
| lowrank16 | 0.953086 | 0.019275 | -0.008025 | 0.504019 |
| fixed_mean | 0.969753 | 0.014384 | +0.008642 | 0.590706 |
| fixed_sqrt_sum | **0.974074** | 0.009259 | +0.012963 | 0.0782023 |
| fixed_sum | 0.973457 | 0.009135 | +0.012346 | 0.170973 |

Fixed scale-only differences after normalization:

```text
fixed_sqrt_sum vs fixed_mean   +0.004321   p=0.779848
fixed_sum      vs fixed_mean   +0.003704   p=0.800000
```

### Interpretation

Once positive scale is controlled, the dramatic fixed-receiver difference
vanishes. Dense, grouped, and fixed local aggregation all occupy essentially
the same accuracy band on this small task. No candidate earns a statistically
credible accuracy advantage from this three-seed audit.

This is the clean capacity-side result Y needed:

> **At this scale, a narrow receiver with parameter-free local nonlinear
> aggregation does not require a rich learned reducer to recover the dense
> bottleneck's task accuracy once optimization scale is controlled.**

That is compatible with the motivating paper's claim that dendritic/local
aggregation need not buy unique learning capacity; the proposed engineering
value is communication/locality, not magical expressivity.

---

# What this corrects

The following old sentence is withdrawn as a general conclusion:

> fixed dendritic pooling loses to an ordinary learned bottleneck

A narrower accurate statement is now:

> **Historical Y fixed-*mean* blocks optimized badly in unnormalized Gate 0/1.
> That failure cannot be interpreted as evidence that fixed local aggregation
> lacks capacity.**

Gate 1's sparse frontier result remains useful only for its exact tested object:
a zero-initialized learned correction growing on top of the historical
fixed-mean receiver. It should not be generalized into a theorem that learned
pre-collapse reduction must consume 50% of the budget.

---

# Gate 2 verdict so far

```text
unique learned receiver needed for small-task capacity       NO
interior grouped receiver established as accuracy winner     NO
low-rank receiver established as winner                      NO
historical fixed-branch capacity null still valid            NO
hardware/locality advantage demonstrated                     NOT YET
```

So Y returns to the reason it was created:

> **Can rich local computation be aggregated before an expensive boundary so
> the wide local state does not have to move through the machine?**

The next experiment is not another Digits architecture sweep. It is an actual
CUDA systems gate comparing dense, fixed-local, grouped, and low-rank controls
while keeping receiver width and learned budget explicit.

The reference PyTorch implementation may materialize the wide local state and
therefore may fail to realize any locality advantage. That is not a reason to
rename logical width as hardware savings. Eager, compiler-fused, allocation,
kernel, and physical-memory-traffic measurements must remain separate.

Physical DRAM-byte or energy claims require hardware counters / an appropriate
profiler; PyTorch allocation numbers alone are insufficient.
