# Y — CURRENT HANDOFF

**Updated:** 2026-08-16  
**Status:** Gate 0/1 interpretation corrected; Gate 2 accuracy/fairness audit completed; hardware/locality gate open.

## Read first

1. `docs/GATE2_FAIRNESS_RECEIPT.md` — **current scientific endpoint**: the old fixed-mean null was scale-confounded; after scale control dense/grouped/fixed receivers occupy the same small-task accuracy band.
2. `docs/GATE2_COST_LOCALITY_PROTOCOL.md` — systems gate and stop conditions.
3. `docs/GATE1_PRECOLLAPSE_FRONTIER_RECEIPT.md` — historical sparse-correction frontier; retain, but read with the correction below.
4. `docs/GATE0_FIRST_RECEIPT.md` — corrected minibatch-paired synthetic gate; fixed-mean capacity interpretation is superseded.
5. `experiments/gate2_fairness_audit.py` — scale/initialization audit.
6. `experiments/gate2_cost_locality.py` — accuracy + CUDA reference harness.
7. `y/efficient.py` — Gate-2 dense, grouped, low-rank, and fixed receiver controls.

---

## One-line state

> **Y has not found a special learned receiver. More importantly, it discovered that the earlier fixed-branch failure was mostly an optimization-scale artifact: once receiver scale is controlled, fixed local nonlinear aggregation can match the ordinary dense bottleneck on Digits. The open question is now the one the project should have been asking all along — whether that local aggregation actually reduces hardware communication/memory cost.**

---

# Critical correction to Gate 0 / Gate 1

Historical Y branch blocks used:

```text
receiver = mean_k ReLU(branch_k)
```

The motivating Wu-style formal block uses a **sum** over nonlinear branch outputs.

For fixed K,

```text
sum = K * mean
```

so these have the same connectivity and representational family. In an
unnormalized deep MLP, however, that constant scale strongly changes training.

Gate 2 measured the effect directly.

Three paired Digits splits, 15 epochs, R=32, K=16:

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

The same fixed architecture went from catastrophic to strong when only its
constant aggregation scale changed. Therefore the old sentence

```text
fixed dendritic pooling loses to an ordinary learned bottleneck
```

is **withdrawn as a capacity conclusion**.

A precise historical statement is still allowed:

```text
fixed-mean aggregation optimized badly in the old unnormalized Gate 0/1 setup
```

---

# Scale-controlled capacity audit

Gate 2 then inserted a parameter-free
`LayerNorm(R, elementwise_affine=False)` after each hidden receiver. This adds
no learned parameters and largely removes positive constant receiver scale as a
confound.

Three paired Digits splits, 15 epochs:

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

Paired differences versus dense were not significant in this n=3 audit.
Among the three identical-connectivity fixed variants, normalization collapsed
the scale gap:

```text
fixed_sqrt_sum vs mean   +0.004321   p=0.779848
fixed_sum      vs mean   +0.003704   p=0.800000
```

Current capacity verdict:

```text
special learned reducer required       NO
fixed local aggregation capacity null  NO — old null was confounded
interior grouped winner established    NO
dense bottleneck uniquely superior     NO
```

This small-task result is compatible with the motivating paper's core framing:
local nonlinear aggregation need not be a capacity trick. Its interesting
engineering value would have to come from locality / communication.

---

# Gate 1 is retained, but narrowed

Gate 1's exact object was:

```text
wide local ReLU state
-> fixed grouped-MEAN base receiver
-> zero-initialized sparse learned correction
```

The correction weights were deliberately initialized to zero. The dense
bottleneck reducer was normally/randomly initialized.

Therefore Gate 1 cleanly tested:

> **Can a cheap sparse learned correction grow out of the historical fixed-mean receiver and match the dense bottleneck?**

On ten paired Digits splits the answer was no below the 50% reducer-budget end.
That result remains banked.

It did **not** cleanly establish:

> every sparse/local learned reducer is intrinsically worse than dense reduction

Do not overgeneralize the receipt.

---

# NEXT GATE — actual hardware locality

Do **not** return to routing, growth, WidePresent, receiver banks, or another
neuron activation yet.

The next question is:

> **At matched receiver width and explicit learned budget, can fixed/local or
> structured reduction reduce measured hardware cost without giving back the
> capacity that the normalized audit preserved?**

Shortlist:

```text
dense bottleneck        ordinary control
fixed local aggregation paper-form/topology control
grouped2                mild learned-local reducer
grouped4                stronger learned-local reducer
lowrank16               standard factorized control
```

Keep these quantities separate:

```text
logical receiver width
parameter count
FLOPs
wide local activation width
whether that activation is materialized
PyTorch allocated memory
kernel count / launch overhead
wall-clock latency
physical DRAM traffic
energy
```

A reference eager PyTorch block can have a narrow public interface while still
materializing the entire wide local state in global memory. That does **not**
earn a communication claim.

Run the existing reference CUDA gate locally with:

```bash
python experiments/gate2_cost_locality.py --hardware-only --bench-backward
python experiments/gate2_cost_locality.py --hardware-only --bench-backward --bench-dtype float16
python experiments/gate2_cost_locality.py --hardware-only --compile
```

The existing harness still includes a historical fixed-mean name in parts of
its sweep; use `experiments/gate2_fairness_audit.py` for scale science. The
next code task is to freeze a **hardware shortlist** whose fixed candidate uses
paper-form/local-sum topology explicitly and whose timing cannot overflow from
chaining raw sums through many blocks.

Physical DRAM-byte claims require hardware counters / an appropriate profiler;
peak allocation alone is insufficient.

---

## Stop conditions

```text
If eager/compiled dense is cheaper and no local implementation closes gap:
    no Y efficiency block yet.

If low rank spans the best accuracy/cost frontier:
    established method explains the win; do not claim Y novelty.

If grouped/fixed local topology preserves accuracy but PyTorch materialization kills cost:
    the next valid task is a fused/local kernel replication, not routing.

If a fused/local implementation gives a robust wall-clock/memory advantage:
    replicate on a second workload before reopening context-dependent receivers.
```

No hardware-efficiency claim has been earned yet.
