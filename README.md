# Y

**Communication-bounded neural computation.**

> **Compute locally. Communicate only what the next stage can use. Change the receiver only when the current receiver is information-limited.**

Y is the pivot from the GeometricNeuron / Dig / PivotPoint / WidePresent trail
toward a concrete AI-efficiency question.

It is **not** a claim that dendrites make neural networks magically more
expressive. Wu et al. (2026), *Dendritic nonlinearities mitigate communication
costs*, report the more useful result: when model complexity is controlled,
active-dendrite-style networks can retain comparable learning capacity while
localized nonlinear aggregation reduces the width communicated onward.

Y starts from that engineering observation and tries to falsify the easy story
before adding anything fancy.

**Provenance note (2026-08-16):** Y has not verified an author-linked public
implementation of the 2026 paper or a published fused Triton kernel. Earlier Y
docs accidentally stated that such code had already been verified; that claim
has been removed. Any fused/local kernel implemented here is a Y systems
replication unless an author-linked source is explicitly identified later.

---

## The object

```text
narrow receiver x
       |
       v
   rich local compute
       |
 fixed / learned local aggregation
       |
       v
narrow receiver y
```

The question is not whether the diagram is narrow. The question is whether the
wide local state can **actually remain local** instead of being materialized,
written to memory, read back, and merely hidden behind a narrow Python API.

For receiver width `R` and hidden budget factor `K`, Y uses the explicit learned
weight budget

```text
B = K * R^2
```

and separates:

```text
logical receiver width
learned parameters / MACs
wide local activation width
PyTorch allocation
kernel behavior
wall clock
physical memory traffic
energy
```

Those are not interchangeable measurements.

---

# Gate 0 — synthetic receiver sanity check

The original exploratory run was not strictly minibatch-paired across
architectures. The corrected ten-seed quarter-width synthetic means were:

```text
fixed-mean branch R=32   0.65015
ordinary bottleneck      0.71748
post-collapse mixer      0.46309
```

What still survives:

- quarter-width communication was not intrinsically impossible;
- a linear mixer **after** collapse was a poor use of the same budget;
- experimental pairing matters.

What no longer survives as a general conclusion is “fixed branch aggregation
lacks capacity.” Gate 2 later showed that the historical branch `mean` was a
major optimization-scale confound.

See `docs/GATE0_FIRST_RECEIPT.md` and the correction in
`docs/GATE2_FAIRNESS_RECEIPT.md`.

---

# Gate 1 — pre-collapse learned receiver frontier

At fixed `R=32`, `K=16`, Gate 1 moved learned budget from local feature creation
into a sparse learned correction before collapse:

```text
H = K*R - s
R*H + R*s = K*R^2
```

Ten paired Digits splits / 40 epochs produced:

| model | reducer budget | local width | mean accuracy |
|---|---:|---:|---:|
| point D=128 | — | 128 | 0.97630 |
| ordinary bottleneck | 50% | 256 | 0.96241 |
| pre-collapse s=256 | 50% | 256 | 0.96056 |
| pre-collapse s=128 | 25% | 384 | 0.94426 |
| pre-collapse s=64 | 12.5% | 448 | 0.93074 |
| pre-collapse s=32 | 6.25% | 480 | 0.92704 |
| historical fixed-mean branch | 0% | 512 | 0.57333 |

The interior sparse correction did **not** beat the dense bottleneck on the
first external dataset.

But Gate 1's interpretation is now deliberately narrow: its sparse correction
was zero-initialized on top of the historical fixed-mean receiver. It answers
whether a cheap correction can grow out of that starting point. It does **not**
prove that all sparse/local learned receivers intrinsically need 50% of the
budget.

See `docs/GATE1_PRECOLLAPSE_FRONTIER_RECEIPT.md`.

---

# Gate 2A — the scale correction

This is the current capacity-side scientific checkpoint.

The motivating fixed local block aggregates nonlinear branch outputs by **sum**.
Historical Y used **mean**. With fixed K those have identical connectivity and
differ only by a constant positive scale — but in an unnormalized deep MLP the
training behavior was radically different.

Three paired Digits splits / 15 epochs, unnormalized:

```text
dense             0.921605
fixed_mean        0.409877
fixed_sqrt_sum    0.922222
fixed_sum         0.961728
```

That enormous gap cannot be a representational-capacity difference: the fixed
variants have the same learned weights and connectivity.

Gate 2 then inserted parameter-free LayerNorm after every hidden receiver to
remove positive scale as a confound. Means became:

```text
dense             0.961111
grouped2          0.962963
grouped4          0.973457
lowrank4          0.947531
lowrank8          0.954321
lowrank16         0.953086
fixed_mean        0.969753
fixed_sqrt_sum    0.974074
fixed_sum         0.973457
```

At this small n, no candidate established a credible accuracy advantage. More
importantly, fixed local aggregation was no longer inferior to the dense
bottleneck once scale was controlled.

So Y withdraws the old sentence:

> fixed dendritic pooling loses to an ordinary learned bottleneck

and replaces it with:

> **historical fixed-mean Y blocks optimized badly in the unnormalized Gate
> 0/1 setup; that is not evidence that fixed local aggregation lacks capacity.**

See `docs/GATE2_FAIRNESS_RECEIPT.md`.

---

# Gate 2B — hardware/locality: OPEN

This is now the serious experiment.

Capacity-side Y has converged back toward the motivating result: rich local
nonlinear computation can feed a narrow receiver without requiring a special
learned reducer on this toy. What remains unearned is the thing that matters:

> **Does the wide local state actually cost less to communicate / access on a
> real GPU?**

The current CUDA instrument is:

```text
experiments/gate2_hardware_shortlist.py
```

and it deliberately separates **two different questions**.

## Paper replication axis

```text
wide_point  vs  fixed_sum
```

For narrow receiver `R` and budget factor `K`, define

```text
D = R * sqrt(K)
```

Then

```text
wide_point: D -> D
fixed_sum : R -> K*R -> grouped nonlinear SUM -> R
```

Both have exactly `K*R^2` learned weights / learned MACs per sample. For
`K=16`, `fixed_sum` exposes one quarter as many boundary activations.

If `fixed_sum` wins here, Y has reproduced the communication geometry motivating
the paper. **That is replication, not Y novelty.**

## Y practical axis

```text
dense narrow bottleneck  vs  fixed_sum
```

The ordinary dense bottleneck already exposes the same R-wide boundary and uses
the same learned budget:

```text
dense: R -> K*R/2 -> R
fixed: R -> K*R   -> deterministic group sum -> R
```

This is the harder Y question:

> **When a boring bottleneck already communicates only R values, is the
> deterministic local collapse actually cheaper to realize?**

If the dense bottleneck is as fast / memory-friendly, the dendritic framing has
not given Y a useful block even if the paper axis succeeds.

Standard controls remain:

```text
grouped2
grouped4
lowrank16
```

---

## Why the size sweep matters

The motivating GPU analysis is cache-dependent. Before exploiting L2 reuse,
the equal-compute point and local shapes do not automatically reduce the main
matrix reads; the stronger predicted read benefit appears when the changed
shape enables better cache/block reuse.

So the default experiment sweeps:

```text
R = 256, 512, 1024
K = 16
```

which gives paper-axis point widths:

```text
D = 1024, 2048, 4096
```

and batches:

```text
32, 128, 512
```

A single small-matrix latency number is not a verdict.

Run the reference CUDA sweep:

```bash
python experiments/gate2_hardware_shortlist.py
```

Then training/backward cost:

```bash
python experiments/gate2_hardware_shortlist.py --backward
```

FP16 separately:

```bash
python experiments/gate2_hardware_shortlist.py --dtype float16
```

Scale-controlled stacks:

```bash
python experiments/gate2_hardware_shortlist.py --modes stack --receiver-widths 256 512
```

Optional compiler comparison:

```bash
python experiments/gate2_hardware_shortlist.py --compile
```

Keep eager and compiled tables separate.

The script measures CUDA wall clock and PyTorch active-allocation peaks. It does
**not** call those values physical DRAM traffic. Hardware counters / an
appropriate profiler are required for that claim.

---

## The actual Y systems hypothesis is fusion

A naive fixed block may legitimately be worse than the dense bottleneck.

At the same learned budget, the dense bottleneck has hidden width

```text
K*R/2
```

while a naive fixed block creates

```text
K*R
```

branch activations before summing them. If ordinary PyTorch globally
materializes that branch tensor, the temporary state is twice as wide as the
dense bottleneck's hidden state.

So Y gets no credit for returning only R values if it first writes `K*R` values
to global memory.

The useful local implementation would approximate:

```text
GEMM tile
 -> activation
 -> deterministic K-way reduction
 -> write only R values
```

without globally materializing the branch state.

If profiling says materialization is the bottleneck, a fused local-collapse
kernel is a valid next replication step. But it must then be compared against
an **optimized/fused ordinary MLP/bottleneck**, not merely eager PyTorch.

If the optimized MLP matches or beats it, Y's candidate is occupied / killed.

See `docs/GATE2_GPU_ACCOUNTING.md` and
`docs/GATE2_COST_LOCALITY_PROTOCOL.md`.

---

## What Y is not allowed to claim yet

- receiver width is **not** measured memory traffic;
- equal learned weights are **not** equal hardware cost;
- a wide hidden tensor hidden inside a module is **not** local merely because
  the module returns only R numbers;
- a sparse gather is **not** an efficient kernel;
- fixed/grouped aggregation matching Digits is **not** a novel architecture;
- reproducing `wide_point > fixed_sum` would be a paper replication, not Y
  novelty;
- no Y hardware-efficiency win has been demonstrated.

---

## Roadmap

1. **Gate 0:** preserve the paired synthetic receipt, but treat the old
   fixed-mean capacity interpretation as superseded.
2. **Gate 1:** preserve the sparse-correction null with its narrowed scope.
3. **Gate 2A:** completed — scale control removes the apparent need for a rich
   learned reducer on the small Digits test.
4. **Gate 2B:** run both the paper replication axis and the harder narrow-dense
   control over a size/cache sweep. If materialization kills locality, profile
   before writing a fused local kernel.
5. **Gate 3 — receiver bank:** only if Gate 2B exposes a real cost/accuracy
   frontier not already explained by ordinary efficient layers.
6. **Gate 4 — escalate when blind / structural consolidation:** only after the
   static efficiency story survives.

Do **not** invent routing yet.

Run tests:

```bash
pip install -e .[dev,experiments]
pytest -q
```

Read `HANDOFF_CURRENT.md` first when resuming the project.
