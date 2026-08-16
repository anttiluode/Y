# Y — founding boundary

Y starts from an engineering question, not a biological claim:

> **How much rich local computation can we keep behind a small communication interface, and when should that interface change?**

## What is borrowed

Wu et al. (2026), *Dendritic nonlinearities mitigate communication costs*, supplies the first hard baseline: expand into several local nonlinear branch computations, reduce them before the next layer, and compare at controlled parameter / compute complexity. Their paper and public code already cover fixed branch reduction, communication-cost analysis, ResNet/CIFAR/ImageNet/speech experiments, transformer feed-forward substitutions, and a fused Triton branched-matmul kernel.

So none of those are Y novelty claims.

## What the Antti/Sol repo family contributes

The useful survivors are narrower than the original geometric-neuron story:

- `GeometricNeuronV23`: distinguish **transfer geometry** from **access geometry**; repeated exact-address x STP shuffle gates were null. Morphology may matter by changing which inputs are reachable, not by magical passive transfer.
- `Dig`: a readout `C` imposes a receiver-relative discrimination ceiling. Waiting can accumulate evidence under fixed `C`; changing `C` can expose distinctions the current receiver cannot recover.
- `PivotPoint`: actions should be judged by what they make readable next. Route/probe/wait are operational choices, not metaphysics.
- `WidePresent`: deterministic bookkeeping / boring baselines come before a fancy architecture; preserve nulls.
- `PresentMoment`: heterogeneous local state can be useful, but Y imports no body/brain claim. If temporal state enters Y later, it must solve a measured failure.

## Working principle

**Compute locally. Communicate only what the next stage can use. Change the receiver only when the current receiver is information-limited.**

The first clause is already strongly occupied by the dendritic-communication paper and many bottleneck architectures. The second and third clauses overlap task-aware compression, dynamic-width networks, routing, MoE, active feature acquisition and observability. Y therefore starts as an experimental intersection, not a novelty announcement.

## Hypothesis ladder

### H0 — fixed local aggregation

At matched hidden-core parameter complexity, can a branch-reduce network preserve useful task performance while shrinking block-to-block receiver width?

This is a replication / sanity gate. If it fails in our implementation, stop and debug before doing anything clever.

### H1 — receiver choice at fixed bandwidth

A rich local state may admit several low-dimensional readouts. Can a small context-dependent receiver choose *which* narrow projection to expose, keeping transmitted width fixed while outperforming one universal bottleneck?

Mandatory controls: static wide, static narrow, random receiver, learned static receiver, dynamic-width/slimmable baseline, and equal expected communicated scalars.

### H2 — open another receiver only when blind

If one receiver is insufficient for a specific distinction, can the system detect that failure and pay for a second readout only on those cases?

The trigger must be tied to an externally measurable residual / discrimination failure, not merely a learned `hardness` score renamed with neuroscience language.

### H3 — consolidate repeatedly useful routes

Only after H1/H2: in a nonstationary stream, does repeatedly opening the same extra readout justify making it a persistent local connection? Compare against dynamic sparse training, random regrowth, magnitude pruning, and ordinary architecture growth.

## Efficiency claims: hard rule

Y keeps three numbers separate:

1. **logical receiver width** — how many values cross a modeled block boundary;
2. **PyTorch allocation / latency** — what the reference implementation actually does;
3. **measured hardware traffic** — profiler counters from a fused/local implementation.

A smaller tensor in a diagram is not evidence of lower DRAM traffic. The paper authors already demonstrate why fusion/locality matters. Y cannot claim hardware efficiency until profiler evidence exists on the same workload.

## Immediate stop conditions

- If H0 loses badly at matched core budget, do not decorate it with routing.
- If a standard bottleneck or low-rank layer gives the same accuracy/traffic frontier, dendritic language buys nothing.
- If dynamic receiver selection does no better than Dynamic Slimmable / ordinary conditional computation at equal traffic, call it occupied and move on.
- If hardware profiling shows no real data-movement or latency win, do not market logical width as speed.
