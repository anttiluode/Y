# Boundary-computation archaeology

**Date:** 2026-08-16  
**Status:** synthesis note, not a new positive result  
**Rule:** this note does not reopen routing or architecture search before the Gate 2 hardware gate resolves.

## Why this note exists

The current Y hardware question is whether rich local computation can be collapsed before an expensive communication boundary, rather than materializing and transmitting the whole local state.

A sweep through older PerceptionLab repositories found that this exact *division of labor* has appeared repeatedly under different names. That recurrence is worth recording, but it is not evidence that the mechanism is novel or efficient.

The useful common abstraction is:

```text
rich local state / dynamics
        |
        v
receiver-relevant projection or collapse
        |
        v
narrow carrier
        |
        v
optional temporal eventization
        |
        v
next local domain
```

The stronger engineering form is therefore:

> **Do not export the local bulk. Export only a receiver-sufficient consequence of it, and only as often as that consequence materially changes.**

That is a hypothesis to test, not a conclusion.

---

## 1. GeometricNeuronPlusField: the closest exact ancestor

`GeometricNeuronPlusField/OPERATOR_FLOW_EVENT.md` explicitly proposed:

```text
high-dimensional global mode state
              |
              v
      low-dimensional local mixture
              |
              v
       active temporal boundary
              |
              v
            event
```

It named the last operation **eventization**: converting a distributed continuous field trajectory into sparse, time-addressable output events.

This is almost Y's current spatial-boundary picture with an extra temporal sparsification stage.

Important negative result from that lineage: the particular HH-like nonlinear event boundary did **not** earn computational privilege. Matched linearized and memoryless controls often preserved timing/frequency information better. That is directly relevant to Y: a fancy boundary is not valuable merely because it is nonlinear or biologically evocative. It must beat the boring control.

**Y translation:**

```text
field/modal bulk            -> local hidden computation
soma/local mixture          -> narrow carrier
AIS/event boundary          -> optional sparse/delta transmission
matched linear control      -> dense/fused bottleneck baseline
```

Repository: https://github.com/anttiluode/GeometricNeuronPlusField

---

## 2. GeometricNeuronV5 / RecurrentGeometricNet: temporal communication economy

`GeometricNeuronV5` measured a recurrent held-field regime with:

- approximately `64x` field-velocity silence between held-state and transition regimes;
- approximately `83x` lower maintenance spike rate during dwell than transition;
- a sparse delta-code in which most communication occurs when represented content changes.

The repo's wattage model is assigned rather than measured physical Joules, so the energy interpretation must remain qualified. The temporal sparsity itself is an actual code-level property of the toy dynamics.

`RecurrentGeometricNet` later retained only a weak ~`2x` version after making the reader feed-forward and explicitly identified the missing recurrent field as the likely reason.

**Y translation:** carrier width and carrier update rate are separate axes.

```text
spatial cost   ~ number of values per boundary message

temporal cost  ~ number of boundary messages / changed values over time
```

A future Y block could therefore be evaluated on both:

```text
bytes / token or sample
bytes / second or sequence
```

But temporal eventization must pay for indices, thresholds, metadata, cache state, and worst-case dense-change regimes.

Repositories:

- https://github.com/anttiluode/GeometricNeuronV5
- https://github.com/anttiluode/RecurrentGeometricNet

---

## 3. Resonant-Cortex-Dynamic-Reservoir-Bridge: rich reservoir, tiny exported control state

The reservoir bridge contains `N` complex crystal node states and nonlinear temporal dynamics, then projects that full state onto graph-Laplacian eigenmodes and finally reduces those mode amplitudes to four exported spectral gains:

```text
large crystal state z_k(t)
        |
        v
modal coefficients a_m(t)
        |
        v
STRUCT / MESO / MICRO / DETAIL gains
        |
        v
next stage
```

That was built for an image-generation bridge, not for distributed hardware efficiency. Still, architecturally it is a clean instance of **compute richly inside an island, expose only a compact control surface**.

This suggests a possible receiver-aware Y reducer later: a local field/reservoir need not reconstruct or transmit all of its internal coordinates if the next stage only consumes a small set of task-relevant modes.

That idea is frozen until the current hardware gate resolves.

Repository: https://github.com/anttiluode/Resonant-Cortex-Dynamic-Reservoir-Bridge

---

## 4. ShadyStuff spectral islands: compressed coordinates between islands

`spectral_islands_field_backprop.md` described a network as local spectral islands linked by projection bridges, with each island communicating compressed spectral coordinates downstream rather than its complete local basis/state.

That document contains broader biological/Koopman claims that are not established by this archaeology pass. The useful surviving architectural sentence is narrower:

```text
local rich representation
    -> projection bridge
    -> compressed coordinate
    -> next local representation
```

This is a conceptual precursor, not evidence.

Repository: https://github.com/anttiluode/ShadyStuff

---

## 5. TransientWaveCompiler: the port / identifiability warning

TWC is the strongest systems-theory warning against over-compressing the boundary.

Its reciprocal systems can have rich internal sparse state while exposing only a small number of physical ports (`S11`, `S21`). The project repeatedly found that a small external response can be insufficient to identify the literal internal realization: multiple internal topologies may be response-equivalent.

That gives Y an important counterpart to the communication objective:

> **A narrow carrier is useful only while it preserves the distinctions the receiver needs.**

The interface can be narrow because the downstream task is narrow. It cannot be narrow by decree.

This is almost the same stop rule already present in Y/Dig language: expand/change the receiver only after showing the present receiver is information-limited.

Repository: https://github.com/anttiluode/TransientWaveCompiler

---

## 6. deerskin-hypothesis audit: readout geometry can determine the regime

The later audit of the original checkerboard loop found that changing which taps were read could change the qualitative dynamical regime: a tested 64-resolution configuration was dead with four taps and alive with eleven, in both the simulator and the physical PerceptionLab instrument.

The old speculative story around that system was heavily corrected; the useful audited point is simply:

> **the readout/interface geometry can matter as much as the bulk dynamics.**

For Y this is another reason not to optimize only carrier width. Which coordinates are exposed, and what the receiver can distinguish from them, matters.

Repository: https://github.com/anttiluode/deerskin-hypothesis

---

## 7. Rajapinta and Sigh: useful metaphors, weaker evidence

`RajapintaFable` and `ConnesClockfieldRajapinta` repeatedly cast the boundary as the place where a larger evolving/frozen field is sampled into an observable. This is a strong naming/metaphorical resonance with Y, but the speculative physics does not provide evidence for the hardware claim.

`BrainSighMachine` similarly maps a large EEG/image field through spectral filtering into a smaller set of highlighted structures/peaks. Again, this is a motif, not a communication benchmark.

Use these for intuition only. Do not cite them as proof that Y's systems mechanism works.

Repositories:

- https://github.com/anttiluode/RajapintaFable
- https://github.com/anttiluode/ConnesClockfieldRajapinta
- https://github.com/anttiluode/BrainSighMachine

---

## 8. KYY supplies the methodological control

KYY's local reciprocal scattering model could solve state-tracking tasks, but a generic Householder-product baseline could also solve them and in some cases did so with smaller state. KYY therefore refused to claim that geometry was special.

Y needs the same discipline:

```text
fixed/local collapse
    must beat
optimized ordinary bottleneck / low-rank / grouped / quantized controls
```

If a standard mechanism occupies the same Pareto point, the result is useful engineering but not a special Y primitive.

Repository: https://github.com/anttiluode/KYY

---

## 9. External prior-art checkpoint: the broad application is occupied

The broad statement

> "compress activations at pipeline boundaries to reduce distributed-training communication"

is already an active research area.

A particularly close current example is Janson, Oyallon & Belilovsky (2026), *Learned Subspace Compression for Communication-Efficient Pipeline Parallelism* (arXiv:2606.05484). It treats inter-stage boundary activations as low-rank, learns stage-specific orthogonal projectors, and reports compression/performance Pareto results including aggressive compression regimes.

Older model-parallel work also studies activation/gradient quantization and sparsification, and split-computing work compresses intermediate features for edge/cloud inference.

Therefore Y must **not** claim activation compression or narrow pipeline carriers as new.

The potentially unoccupied combination worth testing is narrower:

1. **architect the local computation itself around the communication boundary**, rather than merely compressing a conventional activation after it has been produced;
2. **fuse local nonlinear expansion + collapse so the expanded state is never materialized outside the local tile/domain**;
3. optionally add **temporal delta/event transmission** so a narrow carrier is also quiet when its receiver-relevant state is unchanged;
4. choose the collapse by **receiver sufficiency**, not reconstruction fidelity of the entire upstream state;
5. compare against optimized/fused standard bottlenecks and standard learned projection/compression methods.

This is the remaining research question, not an earned novelty claim.

---

# Synthesis

Across several repositories built for different reasons, the same decomposition recurs:

```text
                 LOCAL ISLAND
        +--------------------------+
        | rich geometry / field    |
input ->| recurrence / resonance   |
        | nonlinear local compute  |
        +-------------+------------+
                      |
              receiver-sufficient
                  projection
                      |
                narrow carrier
                      |
             [optional delta gate]
                      |
================== expensive boundary ==================
                      |
                 NEXT ISLAND
```

A useful name for the *engineering property*, without claiming a new architecture, is:

> **boundary-conditioned computation** — organize computation so that representational richness can be large locally while the state that must cross the physical boundary remains small and receiver-sufficient.

The key metric is not parameter count alone and not activation width alone. It is a Pareto surface over at least:

```text
task quality
local compute
local temporary memory / DRAM traffic
boundary bytes
boundary message rate
latency
```

The old repos add one genuinely useful idea to current Y: **communication has both a spatial width and a temporal duty cycle.**

The current Gate 2 remains first. Do not implement the modal/eventized version until fixed local collapse has survived the fused-hardware comparison against a strong ordinary bottleneck.
