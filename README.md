# Y

**Communication-bounded neural computation.**

> **Compute locally. Communicate only what the next stage can use. Change the receiver only when the current receiver is information-limited.**

Y is the pivot from the GeometricNeuron / Dig / PivotPoint / WidePresent research trail toward a concrete AI-efficiency question.

It is **not** a claim that dendrites make neural networks more expressive. A 2026 paper by Wu et al., *Dendritic nonlinearities mitigate communication costs*, reports the more useful result: when model complexity is controlled, active-dendrite networks can retain comparable learning capacity while local nonlinear aggregation reduces the width that must be communicated onward. Their public code already includes fixed branch-reduce models and a fused Triton branch-matmul -> ReLU -> reduction kernel.

Y starts **after** that result.

## The object

The first reference block is deliberately simple:

```text
narrow receiver x
       |
       v
   rich local compute
       |
 local compression
       |
       v
narrow receiver y
```

The paper's branch-reduce construction is one candidate local compressor, not an axiom. Y immediately compares it against an ordinary learned bottleneck at the same receiver width and hidden weight budget.

For a square point layer of width `D`, there are roughly `D^2` weights. A K-branch square block with receiver width `R` has roughly `K R^2` weights. Matching hidden-core weight complexity gives

```text
R ~= D / sqrt(K)
```

so `K=4` suggests a half-width receiver and `K=16` a quarter-width receiver while preserving square hidden-core weight count.

That is the first sanity check, not the final architecture.

## Why this repo exists if the paper already did it

Because the earlier repo trail left one useful engineering question behind.

`Dig` found that information is **receiver-relative**: waiting under one readout can accumulate evidence, but it cannot recover distinctions outside that readout's ceiling. `PivotPoint` turned that into an action question: when should a system wait, route, or probe? `GeometricNeuronV23` separately pushed morphology toward **access/search geometry** after passive exact-address stories repeatedly failed strict shuffle tests.

Y asks whether those ideas can become an efficiency primitive:

> Keep rich computation local behind a narrow interface. If the current receiver cannot support the distinction the task needs, change the receiver or briefly pay for another one instead of widening everything all the time.

Dynamic width, routing, task-aware compression and MoE already occupy much of this territory. Y therefore uses kill gates and strong controls rather than treating the framing as novelty.

## Gate 0 — fixed local aggregation

```bash
pip install -e .
python experiments/gate0_branch_reduce.py --quick
```

The first gate compares:

```text
point width D
vs
branch K=4 / K=16
vs
ordinary learned bottleneck at exactly the same narrow receiver width
and the same hidden weight budget
```

The first three-seed synthetic result is already useful:

```text
mean held-out accuracy

point D=128             0.7627   receiver 1.00x
branch K=4, R=64        0.7459   receiver 0.50x
bottleneck K=4, R=64    0.7375   receiver 0.50x
branch K=16, R=32       0.6592   receiver 0.25x
bottleneck K=16, R=32   0.7235   receiver 0.25x
```

The important correction is the last pair: the poor `K=16` branch result does **not** mean a quarter-width receiver is inherently too small. An ordinary learned bottleneck at the same width and parameter count recovers much of the loss. So the first thing surviving is **communication-bounded local computation**, not fixed dendritic branch averaging as a privileged mechanism.

The output also reports exact parameter count and a **logical receiver-traffic proxy**. This proxy is not measured DRAM traffic. Hardware claims require a later fused-kernel profiler gate.

Run tests:

```bash
pytest -q
```

## Roadmap

1. **H0 fixed aggregation** — move branch and ordinary bottleneck controls to a standard external dataset.
2. **H1 receiver bank** — same transmitted width, different local readout chosen by context; compare against ordinary bottlenecks and dynamic-width networks.
3. **H2 escalate when blind** — pay for an additional receiver only when the current one has a measured task-relevant discrimination failure.
4. **H3 structural consolidation** — only if H2 survives: repeated useful routes can become persistent in continual learning.
5. **hardware gate** — fused implementation + profiler counters. No speed/energy claim before this.

See [`docs/FOUNDING.md`](docs/FOUNDING.md) for prior-art boundaries and stop conditions and [`docs/GATE0_FIRST_RECEIPT.md`](docs/GATE0_FIRST_RECEIPT.md) for the first result and its control-driven correction.

## Current status

**First gate partially survived, mechanism demoted.** Narrow interfaces with rich local compute are worth carrying forward. Fixed branch averaging is not yet special, and at high compression the ordinary learned bottleneck is clearly stronger in the first toy. No hardware-efficiency claim has been earned.
