# Y — founding boundary

Y starts from an engineering question, not a biological claim:

> **How much rich local computation can we keep behind a small communication interface, and when should that interface change?**

## What is borrowed

Wu et al. (2026), *Dendritic nonlinearities mitigate communication costs*,
supplies the first hard baseline: expand into several local nonlinear branch
computations, aggregate before the next layer, and compare at controlled model
complexity. The paper reports comparable learning capacity to point-neuron
models when complexity is controlled and argues that localized aggregation can
reduce communication / memory-access overhead.

Those ideas are prior art and are not Y novelty claims.

**Provenance correction, 2026-08-16:** earlier versions of this document stated
that the authors' public code had already been verified to include a fused
Triton branched-matmul kernel. Y has not verified an author-linked public
implementation of the 2026 paper. That implementation claim is withdrawn. If Y
builds a fused/local kernel, treat it as a systems replication unless an
explicit author-linked source is identified.

## What the Antti/Sol repo family contributes

The useful survivors are narrower than the original geometric-neuron story:

- `GeometricNeuronV23`: distinguish **transfer geometry** from **access
  geometry**; repeated exact-address × STP shuffle gates were null. Morphology
  may matter by changing which inputs are reachable, not by magical passive
  transfer.
- `Dig`: a readout `C` imposes a receiver-relative discrimination ceiling.
  Waiting can accumulate evidence under fixed `C`; changing `C` can expose
  distinctions the current receiver cannot recover.
- `PivotPoint`: actions should be judged by what they make readable next.
  Route/probe/wait are operational choices, not metaphysics.
- `WidePresent`: deterministic bookkeeping / boring baselines come before a
  fancy architecture; preserve nulls.
- `PresentMoment`: heterogeneous local state can be useful, but Y imports no
  body/brain claim. If temporal state enters Y later, it must solve a measured
  failure.

## Working principle

**Compute locally. Communicate only what the next stage can use. Change the
receiver only when the current receiver is information-limited.**

The first clause is already strongly occupied by dendritic-communication work,
bottleneck architectures, grouped/structured layers, and hardware-aware local
aggregation. The later clauses overlap task-aware compression, dynamic-width
networks, routing, MoE, active feature acquisition and observability. Y starts
as an experimental intersection, not a novelty announcement.

---

# What the first gates actually established

## Gate 0 — paired synthetic sanity gate

The corrected quarter-width synthetic comparison showed that a narrow receiver
can still learn, and that a post-collapse mixer is a poor use of the same
budget. Historical fixed-mean branches underperformed the ordinary bottleneck.

That last observation is now explicitly **not** treated as a general capacity
result.

## Gate 1 — sparse correction frontier

At fixed receiver width and learned-weight budget, a zero-initialized sparse
pre-collapse correction growing out of the historical fixed-mean receiver did
not match the dense bottleneck until the reducer budget reached the dense
endpoint on Digits.

That remains a valid null for that specific mechanism and initialization. It is
not a theorem about every sparse/local receiver.

## Gate 2A — scale/fairness correction

The fixed branch architecture was audited with three deterministic aggregation
scales:

```text
mean
sum / sqrt(K)
sum
```

They have identical learned weights and connectivity. Without normalization,
training accuracy differed enormously. With parameter-free LayerNorm after each
hidden receiver, the difference collapsed and fixed/grouped/dense models all
occupied the same small-task accuracy band.

So the old conclusion

```text
fixed local aggregation lacks capacity
```

is withdrawn.

The cleaner current conclusion is:

> **At this scale, no rich learned reducer is required to recover dense-
> bottleneck task accuracy once receiver scale is controlled.**

That pushes Y back toward the motivating systems question rather than toward a
more elaborate receiver.

---

# Hypothesis ladder — revised

## H0 — capacity under a narrow receiver: provisionally passed on the toy

Can fixed local nonlinear aggregation preserve useful task accuracy at the same
narrow receiver and learned-weight budget as a dense bottleneck?

After scale control on Digits: **yes, provisionally.** This is a small-task
capacity result, not a hardware result and not a novelty claim.

## H0b — physical locality / communication: OPEN

Can the wide local computation be performed and reduced without paying the same
or larger memory/latency cost elsewhere?

Mandatory controls:

```text
dense bottleneck
fixed local aggregation
grouped/block receiver
low-rank factorization
eager versus compiler-fused execution
```

If the reference graph materializes the wide local state and loses, inspect the
kernel/memory behavior before concluding the topology is useless. A fused local
kernel is a valid replication step only if profiling identifies materialization
as the bottleneck.

## H1 — receiver choice at fixed bandwidth: FROZEN

A rich local state may admit several low-dimensional readouts. Can a small
context-dependent receiver choose *which* narrow projection to expose, keeping
transmitted width fixed while outperforming one universal bottleneck?

Do not run this until H0b produces a real cost/accuracy frontier that ordinary
static efficient layers do not already explain.

Mandatory controls when/if reopened: static wide, static narrow, random
receiver, learned static receiver, dynamic-width/slimmable baseline, and equal
expected communicated scalars.

## H2 — open another receiver only when blind: FROZEN

If one receiver is insufficient for a specific distinction, can the system
detect that failure and pay for a second readout only on those cases?

The trigger must be tied to an externally measurable residual / discrimination
failure, not a learned `hardness` score renamed with neuroscience language.

## H3 — consolidate repeatedly useful routes: FROZEN

Only after H1/H2: in a nonstationary stream, does repeatedly opening the same
extra readout justify making it a persistent local connection? Compare against
dynamic sparse training, random regrowth, magnitude pruning, and ordinary
architecture growth.

---

## Efficiency claims: hard rule

Y keeps these distinct:

1. **logical receiver width** — how many values cross the modeled block boundary;
2. **learned work** — parameters / MACs;
3. **wide local state** — how many intermediate activations exist logically;
4. **PyTorch allocation / latency** — what the reference implementation does;
5. **kernel behavior** — whether local state is fused or materialized;
6. **physical hardware traffic / energy** — counter-based measurement.

A smaller tensor in a diagram is not evidence of lower DRAM traffic.

## Immediate stop conditions

- If fixed/grouped local models preserve accuracy but dense remains cheaper in
  honest eager and compiled measurements, Y has no efficiency block yet.
- If low rank or a standard grouped layer gives the same or better frontier,
  dendritic language buys no new primitive.
- If the only problem is materializing the local branch state, a fused kernel
  replication is allowed before routing.
- If hardware profiling shows no real data-movement or latency advantage after
  an appropriate local implementation, do not market logical width as speed.
- Dynamic receiver selection remains frozen until the static systems gate
  survives.
