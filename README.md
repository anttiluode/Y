# Y

**Communication-bounded neural computation.**

> **Compute locally. Communicate only what the next stage can use. Change the receiver only when the current receiver is information-limited.**

Y is the pivot from the GeometricNeuron / Dig / PivotPoint / WidePresent research trail toward a concrete AI-efficiency question.

It is **not** a claim that dendrites make neural networks more expressive. A 2026 paper by Wu et al., *Dendritic nonlinearities mitigate communication costs*, reports the more useful result: when model complexity is controlled, active-dendrite networks can retain comparable learning capacity while local nonlinear aggregation reduces the width that must be communicated onward. Their public code already includes fixed branch-reduce models and a fused Triton branch-matmul -> ReLU -> reduction kernel.

Y starts **after** that result.

## The object

```text
narrow receiver x
       |
       v
   rich local compute
       |
 learned / fixed compression
       |
       v
narrow receiver y
```

The paper's branch-reduce construction is one candidate compressor, not an axiom. Y compares it against boring controls at the same receiver width and hidden learned-weight budget.

For a square point layer of width `D`, there are roughly `D^2` weights. A K-budget narrow block with receiver width `R` uses

```text
K * R^2 ~= D^2
```

so `K=4` suggests a half-width receiver and `K=16` a quarter-width receiver while preserving the square hidden-core learned-weight count.

## Why this repo exists if the paper already did it

Because the earlier repo trail left one useful engineering question behind.

`Dig` found that information is **receiver-relative**: waiting under one readout can accumulate evidence, but it cannot recover distinctions outside that readout's ceiling. `PivotPoint` turned that into an action question: when should a system wait, route, or probe? `GeometricNeuronV23` separately pushed morphology toward **access/search geometry** after passive exact-address stories repeatedly failed strict shuffle tests.

Y asks whether any of that survives ordinary efficiency controls:

> Keep rich computation local behind a narrow interface. Learn only as much receiver machinery as the task actually needs. If that still loses to an ordinary bottleneck, stop decorating it with biology.

Dynamic width, routing, task-aware compression, structured sparsity and MoE already occupy much of this territory. Y therefore uses kill gates and strong controls rather than treating the framing as novelty.

---

## Gate 0 — fixed aggregation versus ordinary bottleneck

```bash
pip install -e .
python experiments/gate0_branch_reduce.py --quick
```

A methodological correction matters: the first exploratory run did not strictly pair minibatch order across differently shaped architectures. The decisive quarter-width condition was rerun over ten seeds with an explicit identical shuffle generator per model.

Corrected synthetic K=16 result:

```text
branch R=32       0.65015
bottleneck R=32   0.71748
postmix R=32      0.46309
```

So:

- quarter-width communication is not intrinsically impossible;
- the **compression operator matters**;
- a learned linear mixer *after* collapse is a bad use of the same budget;
- fixed dendritic-style averaging is not privileged.

See [`docs/GATE0_FIRST_RECEIPT.md`](docs/GATE0_FIRST_RECEIPT.md).

---

## Gate 1 — pre-collapse receiver frontier

Install the no-download external benchmark dependency:

```bash
pip install -e .[experiments]
python experiments/gate1_precollapse_frontier.py --quick
```

At fixed quarter-width receiver `R=32` and fixed hidden learned-weight budget `K*R^2`, `PrecollapseReceiverBlock` trades feature-generation budget against learned reducer budget exactly:

```text
H = K*R - s

R*H + R*s = K*R^2
```

where `s` is learned reducer fan-in per receiver output.

The committed implementation stores only the active sparse correction weights plus fixed indices. The learned correction touches the **pre-collapse local state**.

Ten paired stratified splits of scikit-learn Digits:

| model | receiver | reducer budget | local width | mean accuracy |
|---|---:|---:|---:|---:|
| point D=128 | 1.00x | — | 128 | **0.97630** |
| ordinary bottleneck | 0.25x | 50% | 256 | **0.96241** |
| pre-collapse s=256 | 0.25x | 50% | 256 | **0.96056** |
| pre-collapse s=128 | 0.25x | 25% | 384 | **0.94426** |
| pre-collapse s=64 | 0.25x | 12.5% | 448 | **0.93074** |
| pre-collapse s=32 | 0.25x | 6.25% | 480 | **0.92704** |
| fixed branch | 0.25x | 0% | 512 | **0.57333** |

**Gate 1 verdict: no interior winner.**

The very sparse receiver matched the dense bottleneck on the synthetic teacher after pairing correction, but that did **not** transfer to Digits. On the first external dataset, learned reduction needs to be rich; the 50% candidate merely ties the ordinary bottleneck.

See [`docs/GATE1_PRECOLLAPSE_FRONTIER_RECEIPT.md`](docs/GATE1_PRECOLLAPSE_FRONTIER_RECEIPT.md).

---

## What Y is *not* allowed to claim yet

- receiver width is **not** measured DRAM traffic;
- equal learned weights are **not** equal hardware cost;
- a sparse reference gather is **not** an efficient kernel;
- a biology-inspired decomposition is **not** an efficiency win;
- no new Y block has been found.

The Wu et al. implementation already demonstrates that fused local branch computation can matter on GPU. Y must beat ordinary efficient-linear controls before making any architecture claim of its own.

---

## Roadmap

1. **Gate 0 — fixed aggregation:** corrected; ordinary bottleneck beats fixed branch at the decisive quarter-width condition.
2. **Gate 1 — pre-collapse frontier:** completed on Digits; **no interior winner**.
3. **Gate 2 — cost/locality:** compare dense bottleneck against low-rank, grouped/block linear, structured sparse and fused-local reducers using real GPU memory/latency measurements. Do not invent routing yet.
4. **Gate 3 — receiver bank:** only if Gate 2 exposes a real cost/accuracy frontier not already solved by ordinary efficient layers, test context-dependent readouts at the same communication budget.
5. **Gate 4 — escalate when blind / structural consolidation:** only after the static efficiency story survives.

Run tests:

```bash
pip install -e .[dev,experiments]
pytest -q
```

See [`docs/FOUNDING.md`](docs/FOUNDING.md) for prior-art boundaries and stop conditions.

## Current status

**The first candidate Y block failed.** That is useful progress.

What remains alive is the broader engineering target: **where should expensive computation live relative to a communication boundary, and what is the cheapest learned receiver that preserves task-relevant distinctions?** The next gate must answer that with ordinary efficient-linear baselines and hardware measurements, not another neuron metaphor.
