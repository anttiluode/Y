# Y — CURRENT HANDOFF

**Updated:** 2026-08-16  
**Status:** Gate 0 corrected; Gate 1 completed; first candidate Y block failed.

## Read first

1. `docs/GATE1_PRECOLLAPSE_FRONTIER_RECEIPT.md` — current endpoint: no interior pre-collapse receiver winner on Digits.
2. `docs/GATE0_FIRST_RECEIPT.md` — corrected synthetic gate and minibatch-pairing methodological repair.
3. `README.md` — project boundary and next gate.
4. `experiments/gate1_precollapse_frontier.py` — reproducible external frontier instrument.
5. `y/models.py` — exact-budget reference blocks.

---

## One-line state

> **Narrow communicated interfaces remain plausible, but fixed dendritic pooling loses to an ordinary learned bottleneck, and moving learned reducer budget below 50% did not preserve dense-bottleneck accuracy on the first external dataset. No Y block has been found.**

---

## Gate 0 correction

The original three-seed experiment was not strictly paired because differently shaped models consumed different amounts of global RNG before one shared shuffled DataLoader sampled minibatches.

Rule now frozen:

```text
same dataset / split
same model seed
fresh DataLoader per architecture
explicit identical shuffle-generator seed
```

Corrected ten-seed synthetic K=16 / R=32 means:

```text
branch        0.65015
bottleneck    0.71748
postmix       0.46309
```

Survivors:

```text
quarter-width receiver is not intrinsically impossible
pre-collapse learned compression matters
post-collapse linear mixing is a bad use of this budget
fixed branch averaging is not privileged
```

Do not use the old K=4 three-seed difference as paired evidence.

---

## Gate 1 object

At receiver width `R`, budget factor `K`, and learned reducer fan-in `s`:

```text
H = K*R - s

feature weights   = R*H
reducer weights   = R*s
--------------------------------
total             = K*R^2 exactly
```

`PrecollapseReceiverBlock` gives every local feature a free grouped-mean path and adds a zero-initialized learned sparse correction that touches the local state before collapse.

For `D=128`, `K=16`, `R=32`:

```text
s=0      reducer 0%       H=512
s=32     reducer 6.25%    H=480
s=64     reducer 12.5%    H=448
s=128    reducer 25%      H=384
s=256    reducer 50%      H=256
```

The ordinary bottleneck is a separate 50/50 endpoint/control with `H=256`.

---

## Gate 1 result — scikit-learn Digits

Ten paired stratified train/test splits, 40 epochs:

```text
point D=128          0.97630
bottleneck R=32      0.96241
pre s=256            0.96056
pre s=128            0.94426
pre s=64             0.93074
pre s=32             0.92704
branch                0.57333
```

Paired versus bottleneck:

```text
s=32      -0.03537   p=0.00336
s=64      -0.03167   p=0.00225
s=128     -0.01815   p=0.00475
s=256     -0.00185   p=0.628
```

Verdict:

```text
interior sparse receiver beats endpoints      NO
cheap learned receiver matches bottleneck     NO on Digits
50% residual receiver beats bottleneck        NO / tie
```

The synthetic corrected ten-seed result had `pre s=32 ~= bottleneck`; that apparent sufficiency did not transfer externally. Preserve this as a warning against tuning synthetic teachers.

---

## Important implementation boundary

The exact sparse reference stores only active correction weights plus fixed integer indices, but its gather implementation is **not** a hardware-efficiency result.

Do not conflate:

```text
logical receiver width
parameter count
FLOP count
activation materialization
DRAM traffic
cache behavior
kernel launch / occupancy
wall-clock
energy
```

These are now separate measurements.

---

# NEXT GATE: cost/locality before routing

Do **not** add:

```text
receiver bank
WAIT/ROUTE controller
structural growth
WidePresent state
more biological nonlinearities
```

just to rescue Gate 1.

The next question is ordinary and hardware-facing:

> **If a richly learned pre-collapse reducer is required anyway, can we make that reducer materially cheaper in communication/locality while preserving the ordinary dense bottleneck's accuracy?**

Mandatory controls:

```text
dense bottleneck
low-rank factorized linear
block/grouped linear
structured sparse linear
fused local branch/reduction where fair
```

Measure on actual GPU:

```text
accuracy
receiver width
learned weights / FLOPs
wall-clock inference + training
peak allocated memory
profiler memory/kernel behavior where available
```

Only if one structured/local design lies on a better measured cost/accuracy frontier than the ordinary dense bottleneck should Y return to context-dependent receiver selection.

## Stop condition

If standard low-rank/grouped/sparse controls span the same or better frontier, then Y's current contribution collapses to a useful framing and negative-results archive. Accept that result rather than inventing another mechanism.
