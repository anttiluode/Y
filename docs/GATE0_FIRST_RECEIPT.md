# Gate 0 — first receipt, with pairing correction

**Date:** 2026-08-16  
**Status:** narrow receiver idea survives; fixed branch reduction is not privileged; post-collapse mixing remains a strong null.

This is a synthetic instrument check, not an external benchmark and not a hardware-efficiency result.

## Important methodological correction

The first three-seed run reused one shuffled `DataLoader` while reseeding before each model. Different architectures initialize differently sized parameter tensors, consuming different amounts of global RNG before the loader begins sampling. Therefore the models saw the same examples but **not strictly identical minibatch orders**.

That does not invalidate large qualitative gaps, but it does invalidate treating small architecture differences in the original table as paired evidence.

The decisive `K=16`, quarter-width condition was rerun over **10 seeds** with a fresh `DataLoader` per model and an explicit identical shuffle generator for every architecture within a seed.

Corrected paired result:

| model | receiver ratio | mean accuracy | std |
|---|---:|---:|---:|
| branch K=16, R=32 | 0.25x | **0.65015** | 0.04483 |
| bottleneck K=16, R=32 | 0.25x | **0.71748** | 0.03598 |
| postmix K=16, R=32 | 0.25x | **0.46309** | 0.10736 |

The branch-versus-bottleneck gap remains large. The post-collapse mixer null also survives strongly.

The old `K=4` three-seed differences have **not** been re-audited with the paired loader. Keep them as historical exploratory numbers only; do not use them as evidence that branch pooling beats an ordinary bottleneck at half width.

## Question

At approximately matched parameter count, can rich local nonlinear computation preserve held-out classification performance while reducing the width communicated between hidden blocks?

The point model uses width `D=128`. Narrow models use receiver width `R` with square hidden-core weight budget

```text
K * R^2 ~= D^2
```

so `K=4 -> R=64` gives `0.50x` logical receiver width and `K=16 -> R=32` gives `0.25x`.

Three narrow mechanisms were compared at the **same receiver width and same hidden weight count**:

```text
BRANCH
    many local ReLU features -> fixed group mean -> R

BOTTLENECK CONTROL
    fewer local ReLU features -> learned pre-collapse compression -> R

POST-COLLAPSE MIXER NULL
    local ReLU branches -> fixed mean -> R -> learned R x R mixer
```

The third design is intentionally tempting: spend one branch-worth of budget on a small learned receiver mixer. But the mixer only sees the already-collapsed receiver, and in a feed-forward stack much of that extra linear map can be absorbed into the next layer. It is therefore a useful negative control for *where* learned compression must act.

## Corrected setup for the decisive audit

```text
input_dim       32
classes          8
hidden depth     4
point width    128
K               16
receiver R      32
train samples 8192
test samples  2048
epochs           30
optimizer       AdamW
seeds          0..9
minibatch order explicit and identical within each seed
```

Labels come from a frozen random nonlinear teacher. Students do not see teacher weights.

## Historical first run — retained, not paired evidence

The original exploration produced:

| model | receiver ratio | params | original mean accuracy |
|---|---:|---:|---:|
| point D=128 | 1.00x | 54,272 | 0.7627 |
| branch K=4, R=64 | 0.50x | 53,760 | 0.7459 |
| bottleneck K=4, R=64 | 0.50x | 53,760 | 0.7375 |
| postmix K=4, R=64 | 0.50x | 53,760 | 0.7223 |
| branch K=16, R=32 | 0.25x | 53,504 | 0.6592 |
| bottleneck K=16, R=32 | 0.25x | 53,504 | 0.7235 |
| postmix K=16, R=32 | 0.25x | 53,504 | 0.4129 |

These values are preserved because the first experiment happened. The corrected audit above supersedes them wherever strict model-to-model comparison matters.

## What survived correction

### 1. Quarter-width is not intrinsically impossible

At the corrected quarter-width condition:

```text
branch K=16 mean       0.65015
bottleneck K=16 mean   0.71748
```

So a large amount of hidden compute can live behind a much narrower interface in this toy; **the compression operator matters**.

### 2. A mixer after collapse is too late / too redundant

Corrected ten-seed postmix mean:

```text
postmix K=16 mean      0.46309
```

This should not be generalized to all mixers. The specific lesson is architectural:

> If Y learns a receiver, the learned map must touch the rich local state **before** the irreversible narrow readout. Spending budget only after that readout cannot restore distinctions it discarded, and a plain linear post-map is largely redundant with the next layer anyway.

This is the useful bridge to the receiver-relative work in `Dig`: changing computation downstream of a fixed readout is not the same thing as changing the readout itself.

### 3. No dendritic privilege has been earned

The corrected decisive condition favors the ordinary learned bottleneck over fixed branch averaging. There is no basis here for claiming a special dendritic mechanism.

## Hardware note

All reference implementations are logical architecture instruments. **Logical receiver width is not hardware efficiency.**

The Wu et al. public implementation already contains a fused Triton branch-matmul -> ReLU -> branch-reduction kernel. Y should not claim a hardware win until a comparable fused/local implementation is profiled on the same workload.

## Decision

**Survives:** communication-bounded local computation as a question.  
**Not earned:** fixed dendritic pooling as the privileged mechanism.  
**Killed:** “K=16 failed, therefore quarter-width is intrinsically too narrow.”  
**Banked null:** post-collapse learned mixing as a good use of the same budget.  
**Methodology rule added:** architecture comparisons must use explicitly paired minibatch order.

The next gate is the **pre-collapse receiver frontier**: keep the receiver width and total hidden learned-weight budget fixed, and trade local feature-generation budget against learned receiver budget before collapse.
