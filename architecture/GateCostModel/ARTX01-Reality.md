# Purpose

This document describes the real, physical execution pipelines that any
future cost model must characterize. It is the factual foundation every
later document in this directory builds on.

---

# Scope

Covers: the glproc (CPU) and glcuda (GPU) execution pipelines, the
physical hardware each depends on, and which parts of GATE's own
architecture (`architecture/GATE/`) are physical process versus pure
software abstraction.

Excludes: measurement methods and units (ARTX02), problem framing
(ARTX03), and any Existing Diagnostic Model's interpretation of these
facts (ARTX04).

---

# Inputs

- `glproc/src/engine.rs`, `glproc/src/threading.rs`,
  `glproc/src/simd_strategy.rs` (glproc pipeline structure).
- `architecture/ArchGLML_X2.md` (glcuda pipeline structure: single VRAM
  allocation, static layer-graph walk, one stream, one sync per token).
- `gl-agent-skills/cpu-skills/memory-bandwidth.md`,
  `gl-agent-skills/cpu-skills/threading-model.md` (physical constraints
  of the CPU reference tier).

---

# Outputs

A description of the execution pipeline that ARTX02's observations are
measurements *of*, and that ARTX09's variables must have a physical
referent *in*.

---

# Requirements

1. The glproc pipeline SHALL be described as: model weights loaded once
   via memory-mapped file access and repacked in parallel at load time;
   per-token decode executed by a static, fixed-size thread pool that
   streams each live weight byte through vector-unit kernels exactly once
   per token, with no reuse across tokens.
2. The glcuda pipeline SHALL be described as: model weights uploaded once
   to a single VRAM allocation at load time; per-token decode executed on
   one command stream with one host-device synchronization point per
   token; prefill executed as batched matrix operations with data reuse
   that decode does not have.
3. The physical substrate for glproc decode SHALL be identified as the
   DDR4 dual-channel memory bus of the reference tier (see
   ARTX06-Terminology.md for the definition of Reference Tier); the
   physical substrate for glcuda SHALL be identified as PCIe transfer and
   on-device VRAM bandwidth.
4. This document MUST distinguish physical operations (memory-bus
   transfer, vector-unit arithmetic, host-device synchronization) from
   software-only abstractions (an `ExecutionPlan`, a `Constraint`, a
   `Policy`) — the latter category consumes no measurable hardware
   resource of its own and MUST NOT be described as if it did.
5. Prefill and decode SHALL be treated as distinct workloads with
   distinct physical characteristics (decode: bandwidth-adjacent,
   serial, no reuse; prefill: batched, compute-adjacent, has reuse) and
   MUST NOT be described by a single, undifferentiated statement about
   "the pipeline."
6. Any statement about a third backend (glvulkan, glmetal) SHALL be
   limited to their documented status as non-functional stubs
   (`available: false` unconditionally) until a real pipeline exists to
   describe.

---

# Non Goals

- This document does not quantify the pipelines (no bandwidth numbers,
  no percentages) — that is ARTX02.
- This document does not judge whether either pipeline is efficient or
  well-architected. It records what exists, not whether it is good.
- This document does not describe glictus-caliburni/GllmEngine's
  pipeline; no comparable reality audit of that engine has been
  performed and none is claimed here.

---

# Exit Criteria

Complete when a reader can state, for glproc and for glcuda, which
resource is consumed by which operation, without consulting any other
document.

---

# References

- ARTX02-Observation.md (quantifies the facts stated here)
- ARTX06-Terminology.md (defines Reference Tier, Backend)
- `architecture/GATE/GATE-mapping.md` (glcore/glproc/glcuda ownership
  boundaries this document's descriptions respect)
