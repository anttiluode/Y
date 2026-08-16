# Gate 1 — pre-collapse receiver frontier

**Date:** 2026-08-16  
**Status:** **NO INTERIOR WINNER** on the first standard external dataset.

This gate asks a narrower engineering question than “do dendrites help?”

> At fixed communicated receiver width and fixed hidden learned-weight budget, how much budget should create private nonlinear features, and how much should learn the receiver that compresses those features *before* communication?

The endpoints are ordinary controls:

```text
fixed branch reducer
    spend almost all learned budget on local features
    parameter-free grouped mean crosses the boundary

ordinary bottleneck
    spend half the hidden budget on local features
    spend half on a dense learned pre-collapse reducer
```

The candidate interior family is `PrecollapseReceiverBlock`.

---

## Exact budget axis

Let

```text
R = communicated receiver width
K = hidden budget factor
s = learned reducer fan-in per receiver output
H = private local feature width
```

Choose

```text
H = K*R - s
```

so each hidden block has exactly

```text
R*H        local feature-generation weights
+ R*s      learned pre-collapse reducer weights
-----------------------------------------------
= K*R^2    total learned hidden weights
```

for every point on the frontier.

All `H` features retain a parameter-free grouped-mean path to the receiver. The learned sparse correction touches the **pre-collapse** local state and is zero-initialized. Its connectivity mask is fixed; this gate learns weights, not topology.

For the decisive quarter-width setting:

```text
point width D       128
K                    16
receiver width R     32        = 0.25x point width
hidden budget     16384 weights per square hidden block
```

The sweep is:

| reducer fan-in `s` | reducer budget | local width `H` |
|---:|---:|---:|
| 0 | 0% | 512 |
| 32 | 6.25% | 480 |
| 64 | 12.5% | 448 |
| 128 | 25% | 384 |
| 256 | 50% | 256 |

The ordinary bottleneck also spends 50% of the hidden budget on its dense reducer with local width 256. It remains a separate control because the `s=256` candidate additionally retains the free grouped-mean path.

---

## Pairing correction carried forward from Gate 0

Every architecture receives its **own** shuffled `DataLoader` with an explicit identical shuffle-generator seed within each split.

This matters because differently shaped models consume different amounts of global RNG during initialization. Reusing an implicitly shuffled loader can therefore give architectures different minibatch orders even after calling the same global seed.

Gate 1 does not use that confounded protocol.

---

## External dataset

`scikit-learn` Digits:

```text
1797 handwritten 8x8 images
64 input features
10 classes
70/30 stratified train/test split
10 split seeds: 0..9
40 epochs
AdamW, lr=2e-3
batch size 64
```

The dataset ships with scikit-learn; no network download is required.

The committed frontier model stores only the active sparse correction weights (`R*s`) plus fixed integer indices. The reference gather implementation is intentionally transparent, **not** a hardware-efficiency claim.

---

## Exact sparse results — 10 paired splits

| model | receiver width | reducer budget | local width | mean accuracy | std |
|---|---:|---:|---:|---:|---:|
| point D=128 | 128 (1.00x) | — | 128 | **0.97630** | 0.00312 |
| bottleneck | 32 (0.25x) | 50% | 256 | **0.96241** | 0.00864 |
| pre-collapse `s=256` | 32 | 50% | 256 | **0.96056** | 0.01059 |
| pre-collapse `s=128` | 32 | 25% | 384 | **0.94426** | 0.01459 |
| pre-collapse `s=64` | 32 | 12.5% | 448 | **0.93074** | 0.02313 |
| pre-collapse `s=32` | 32 | 6.25% | 480 | **0.92704** | 0.02910 |
| fixed branch | 32 | 0% | 512 | **0.57333** | 0.18695 |

Paired differences versus the ordinary bottleneck:

```text
s=32     -0.03537   p=0.00336
s=64     -0.03167   p=0.00225
s=128    -0.01815   p=0.00475
s=256    -0.00185   p=0.62813
branch   -0.38907   p=0.000109
```

The p-values are ordinary paired t-tests across the ten split seeds and are included only as a compact description of this gate, not as a broad generalization claim.

---

## Synthetic re-audit

Before the external run, the corrected ten-seed synthetic quarter-width re-audit gave:

```text
bottleneck        0.71748
pre-collapse s=32 0.71758
pre-collapse s=256 0.71182
fixed branch      0.65015
```

`pre32` was essentially identical to the bottleneck on the toy (`+0.00010`, paired `p≈0.98`).

That result **did not transfer** to Digits. The external gate is therefore doing useful work: the toy made a very sparse learned receiver look sufficient when it was not sufficient on the first real dataset.

---

## Verdict

The preregistered hope was:

> An interior sparse/local learned receiver might beat both fixed branch pooling and an ordinary dense bottleneck at the same transmitted width and learned hidden budget.

**It did not.**

The earned statements are narrower:

1. **Pre-collapse learning matters enormously.** Moving even a small amount of budget into a learned receiver rescues much of the catastrophic fixed-branch loss.
2. **On Digits, cheap learned reduction is not enough.** The 6.25%, 12.5% and 25% receiver-budget points are all significantly below the ordinary bottleneck.
3. **The 50% candidate merely ties the ordinary bottleneck.** The extra parameter-free grouped path does not produce a reliable advantage.
4. **No Y block has been found.** There is no basis for promoting an interior point as a new primitive.

This is a useful null because it prevents the project from drifting into “sparse receiver” enthusiasm based on the synthetic teacher.

---

## What changes next

Do **not** add routing, growth, temporal state or another neuron shape to rescue Gate 1.

The next question is cost-aware and more ordinary:

> If a richly learned pre-collapse reducer is actually required, can its communication/locality cost be reduced **without** giving up the dense bottleneck’s information access?

That requires controls against established efficient-linear mechanisms:

```text
ordinary dense bottleneck
low-rank factorization
block/grouped linear layers
structured sparsity
mixture / conditional computation where appropriate
```

And it requires real measurements:

```text
GPU wall-clock
allocated/peak memory
profiler memory traffic / kernel behavior where available
accuracy at matched receiver width and learned-weight/FLOP budget
```

Until such a gate wins, Y remains a research question about communication boundaries, not a claimed efficient architecture.
