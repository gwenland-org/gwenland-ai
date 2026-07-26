<!--
  GENERATED AGGREGATE — NOT A SPEC DOCUMENT.

  This file is a concatenation of README.md and ARTX00-Research.md through
  ARTX12-Ready.md, in reading order, for a single continuous read or for
  pasting the whole specification elsewhere in one piece.

  It is not part of the ARTX numbering sequence, it is not subject to the
  "one file, one concept" rule or the ~800-word guideline that govern the
  actual specification documents, and it MUST NOT be edited directly. Any
  correction belongs in the source document it was copied from; regenerate
  this file afterward by re-concatenating. If this file and a source ARTX
  document ever disagree, the source ARTX document is authoritative.
-->

# GateCostModel — Full Read

> Concatenation of: README.md, ARTX00-Research.md, ARTX01-Reality.md,
> ARTX02-Observation.md, ARTX03-Problem.md, ARTX04-Theory.md,
> ARTX05-Gap.md, ARTX06-Terminology.md, ARTX07-Ontology.md,
> ARTX08-Assumptions.md, ARTX09-Variables.md, ARTX10-Dependencies.md,
> ARTX11-Validation.md, ARTX12-Ready.md — in that order.

---
---

# SOURCE: README.md

# GateCostModel — Architecture Specification

## Purpose

This directory determines, at the architecture level, whether and how
GwenLand should support ranking not-yet-executed `ExecutionPlan`
candidates (the `architecture/GATE/` protocol's core job) — **before**
any mathematical cost function is written. It exists because a prior
Scientific Foundation Audit of `architecture/GATE/GATE-concepts.md`'s
`MetricVector`/`WeightVector` cost function found the mathematics was
drafted ahead of the architecture that should have constrained it: which
dimensions are even measurable on which backend, which are conditional,
and which of GwenLand's Existing Diagnostic Models already solve part
of the problem. This directory is that missing architecture layer.

This directory does not modify `architecture/GATE/`. It is a sibling
specification that must be read *before* any future revision of
`architecture/GATE/GATE-concepts.md`'s cost-function fields, and it
produces no equations of its own — see ARTX12 for where mathematical work
is directed once the architecture here is stable.

## Reading Order

ARTX00 through ARTX12, in order. Each document depends on conclusions
established in the ones before it; skipping ahead will surface references
to terms and findings not yet introduced.

## Document Map

| Document | Covers |
|---|---|
| ARTX00-Research.md | Why this research effort exists and the rules it follows |
| ARTX01-Reality.md | The real execution pipelines (glproc, glcuda) this specification must describe accurately |
| ARTX02-Observation.md | Measurable facts about those pipelines, with evidence and method |
| ARTX03-Problem.md | The specific engineering problem a cost model would need to solve |
| ARTX04-Theory.md | GwenLand's Existing Diagnostic Models and what each one already does |
| ARTX05-Gap.md | Where existing components fall short of the problem in ARTX03, with evidence |
| ARTX06-Terminology.md | Canonical definitions for every term this document set uses |
| ARTX07-Ontology.md | The structural relationships between those terms |
| ARTX08-Assumptions.md | Assumptions any future cost model would depend on, and their failure conditions |
| ARTX09-Variables.md | Candidate measurable quantities, their physical meaning, and measurability today |
| ARTX10-Dependencies.md | Causal and conditional relationships observed between those quantities |
| ARTX11-Validation.md | What method would be required to confirm or refute each open assumption |
| ARTX12-Ready.md | Architecture readiness verdict and what must happen next |

## Architecture Philosophy

Architecture defines. Mathematics models. Implementation executes. This
directory performs only the first. It states responsibilities,
constraints, relationships, boundaries, and invariants; it does not
propose equations, name variables with mathematical symbols, or specify
algorithms. Where the evidence gathered here implies mathematical work is
needed, that work is named as a follow-on deliverable and left
undone here.

## Contributing Rules

- If this directory already exists at the time of a change, update only
  the affected document(s) — do not rewrite unrelated files.
- Never skip or reuse an ARTX number; never renumber an established
  document.
- Do not introduce a new term without adding it to ARTX06-Terminology.md
  in the same change, and do not redefine a term already canonical in
  `architecture/GATE/` (`ExecutionPlan`, `MetricVector`, `WeightVector`,
  `BackendKind`, `ExecutionPolicy`, `Constraint`, `Validator`) — reuse
  those names as-is.
- No equations, no code, no pseudocode. If a claim cannot be stated
  without one, it belongs in a future mathematical specification, not
  here.
- Where evidence is insufficient, write `UNKNOWN` or `NOT READY`. Do not
  fill a gap with an invented number or an assumed outcome.

---
---

# SOURCE: ARTX00-Research.md

# Purpose

This document establishes why the GateCostModel research effort exists
and the rules it operates under. Every subsequent ARTX document in this
directory inherits these rules; none of them restate it.

---

# Scope

Covers: the origin of this research effort, its governing philosophy, and
the boundary between this directory's architecture work and the
mathematical work it deliberately does not perform.

Excludes: any finding about the execution pipelines, existing analysis
components, or the cost model itself — those belong to ARTX01 onward.

---

# Inputs

- The `architecture/GATE/` specification (`GATE-concepts.md`,
  `GATE-algorithm.md`, `GATE-policy.md`, `GATE-mapping.md`), which defines
  `ExecutionPlan`, `MetricVector`, `WeightVector`, and `ExecutionPolicy`
  as the terms this effort takes as given and does not redefine.
- A prior Scientific Foundation Audit of `GATE-concepts.md`'s cost
  function, which found that dimension selection, measurability, and
  combination method had not been architecturally justified before being
  drafted mathematically.

---

# Outputs

A stated research philosophy and a stated document-responsibility
boundary that every later ARTX document in this directory conforms to.

---

# Requirements

1. This research effort MUST NOT produce equations, symbols, objective
   functions, or proofs. Where mathematical work is found necessary, it
   SHALL be named as a follow-on deliverable for a separate mathematical
   specification (see ARTX12) rather than performed here.
2. This research effort MUST NOT produce code or pseudocode.
3. Every claim in ARTX01 through ARTX11 SHALL be traceable to an
   observation, an existing limitation, or an engineering constraint —
   never to an assumption presented as fact.
4. Where evidence is insufficient to support a claim, the claim MUST be
   recorded as `UNKNOWN` rather than inferred or estimated.
5. Terminology introduced anywhere in this directory MUST be recorded in
   ARTX06-Terminology.md and MUST NOT be redefined elsewhere in the
   directory.
6. This effort MUST NOT alter any file under `architecture/GATE/`. Any
   finding that implies such a file needs revision SHALL be recorded here
   as an open item for a separate, explicitly authorized change.
7. Analysis SHOULD proceed in the order: understand the real system
   (ARTX01–ARTX02), state the actual problem (ARTX03), survey what
   already exists (ARTX04), find the gap (ARTX05), fix vocabulary and
   structure (ARTX06–ARTX07), examine assumptions and variables
   (ARTX08–ARTX10), define how open items get resolved (ARTX11), and only
   then judge readiness (ARTX12). This effort MUST NOT judge readiness
   before completing the preceding steps.

---

# Non Goals

- This effort does not evaluate whether GATE as a whole (the
  generate-validate-evaluate-dispatch protocol) is architecturally sound.
  `architecture/GATE/` already establishes that; only its cost-function
  concern is in scope here.
- This effort does not propose an implementation timeline or a wave plan.
  `architecture/GATE/GATE-impl-plan.md` is the sibling document for that,
  and any consequence for its Wave G3 scope is an output for the user to
  apply there, not a change this directory makes on its own.
- This effort does not re-litigate GATE's correctness proofs (Theorems
  4.2, 10.1–10.3), which concern constraint validation, not cost
  evaluation, and are out of scope for this directory entirely.

---

# Exit Criteria

This document is complete when the research philosophy and document
boundary are stated clearly enough that a reader unfamiliar with the
audit that motivated this directory can determine, from this document
alone, what kind of claim belongs in which later ARTX file.

---

# References

- `architecture/GATE/README.md`
- `architecture/GATE/GATE-concepts.md`
- `architecture/GATE/GATE-mapping.md`
- ARTX06-Terminology.md (term registry this effort must not violate)
- ARTX12-Ready.md (where follow-on mathematical work, if any, is directed)

---
---

# SOURCE: ARTX01-Reality.md

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

---
---

# SOURCE: ARTX02-Observation.md

# Purpose

This document records measurable facts about the pipelines described in
ARTX01, each with its evidence source, measurement method, unit, and
determinism characteristics. It is the evidentiary base for every claim
made in ARTX05, ARTX08, and ARTX09.

---

# Scope

Covers: facts that have been measured or are directly readable from
source, on the Reference Tier, with a named evidence source.

Excludes: interpretation of these facts (ARTX03, ARTX05), and any fact
this directory has not itself re-verified against a current source file
or a current, dated project record.

---

# Inputs

- `glbench/src/environment/bandwidth.rs`,
  `glbench/src/environment/power.rs` (measurement code).
- `glcore/src/telemetry.rs` (measurement vocabulary and units).
- `glbench/src/validation/numerical.rs` (numerical comparison method).
- `gl-agent-skills/cpu-skills/*.md` (dated, measured findings).

---

# Outputs

A table of observations, each traceable to a source, usable without
re-derivation by ARTX05 (Gap), ARTX08 (Assumptions), and ARTX09
(Variables).

---

# Requirements

1. Every observation recorded here MUST cite its evidence source (a file
   path or a named document) and MUST state its unit.
2. Every observation MUST state whether it is deterministic and, if not,
   the source of variance (e.g., thermal state, scheduling noise).
3. The following observations SHALL be recorded as established facts of
   the Reference Tier:
   - Measured multi-threaded sequential memory read bandwidth is
     approximately 29 gigabytes per second, varying by roughly plus or
     minus 19 percent between measurement passes
     (`glbench/src/environment/bandwidth.rs`).
   - A native low-bit-width weight format was measured to execute
     feed-forward kernels at a lower throughput, in multiply-accumulate
     operations per second, than the format currently shipped, and the
     gap persisted when the working set fit in the fastest cache level
     (`gl-agent-skills/cpu-skills/quantization.md`).
   - The same native low-bit-width format was measured to be slower
     end-to-end, by approximately one third, than the shipped format, on
     the same hardware and workload (same source).
   - Feed-forward computation accounts for approximately half of glproc
     decode wall-clock time on the Reference Tier
     (`gl-agent-skills/cpu-skills/memory-bandwidth.md`).
   - Package energy measurement is readable only through a Linux-specific
     kernel interface and is unconditionally unavailable on the Reference
     Tier's host operating system (`glbench/src/environment/power.rs`).
   - Numerical deviation, as currently measured, is a discrete count of
     matching leading tokens against a designated reference engine, valid
     only under fixed-seed, non-sampling decoding
     (`glbench/src/validation/numerical.rs`,
     `glbench/src/validation/deterministic.rs`).
4. An observation not traceable to a source under this document's Inputs
   MUST NOT be recorded here; it belongs in ARTX11 as an item requiring
   validation, or is marked `UNKNOWN`.
5. This document MUST NOT interpret these observations as evidence for or
   against any specific architectural decision — that interpretation is
   ARTX05's responsibility.

---

# Non Goals

- This document does not re-run any measurement; it records the most
  recent dated measurement available in the cited sources.
- This document does not attempt completeness over every possible
  observable quantity — only those bearing on the cost-model question
  this directory exists to answer.
- This document does not measure glcuda's prefill bucket composition
  (attention share versus feed-forward share); prior project records
  reference such a figure, but it was not re-verified against a current
  source during this research effort and is therefore recorded as
  `UNKNOWN` rather than restated from memory.

---

# Exit Criteria

Complete when every observation used by ARTX05, ARTX08, or ARTX09 exists
in this document with a source, a unit, and a determinism statement.

---

# References

- ARTX01-Reality.md (what these observations are measurements of)
- ARTX05-Gap.md (where these observations are interpreted)
- ARTX09-Variables.md (where these observations become named quantities)

---
---

# SOURCE: ARTX03-Problem.md

# Purpose

This document states the specific engineering problem a GATE cost model
would need to solve, rejecting vague framings in favor of a precise
statement traceable to ARTX01 and ARTX02.

---

# Scope

Covers: the precise problem statement, and which of its parts concern
compute, memory, transfer, synchronization, scheduling, or numerical
constraints.

Excludes: what already solves parts of this problem (ARTX04) and where
existing solutions fall short (ARTX05).

---

# Inputs

- ARTX01-Reality.md, ARTX02-Observation.md.
- `architecture/GATE/GATE-algorithm.md` (the generate-validate-evaluate-
  dispatch sequence a cost model would sit inside, at the evaluate step).

---

# Outputs

A precise problem statement that ARTX04 and ARTX05 are evaluated against.

---

# Requirements

1. The problem SHALL be stated precisely as: given more than one
   `ExecutionPlan` candidate for the same computation, each already known
   to satisfy every registered `Constraint`, determine which candidate to
   dispatch, without executing every candidate to find out.
2. This document MUST reject "the system is slow" or "the cost model must
   be accurate" as problem statements; a problem statement SHALL name a
   specific decision point (candidate selection prior to execution) and a
   specific constraint on solving it (without full execution of every
   candidate).
3. This document SHALL identify that the problem is a decision problem,
   not a resource-bound problem: no measurable hardware resource (DRAM
   bandwidth, PCIe transfer, vector-unit throughput) is consumed by the
   decision itself, per ARTX01's distinction between physical operations
   and software-only abstractions.
4. This document SHALL identify the three physical subsystems the
   decision must account for, per ARTX01: DRAM bandwidth (glproc decode),
   vector-unit throughput for weight-unpacking arithmetic (glproc decode,
   backend-specific), and host-device transfer plus stream synchronization
   (glcuda). This document MUST NOT assume these three subsystems are
   comparable across every `BackendKind` without support from ARTX02.
5. Where multiple distinct problems exist (for example: selecting between
   two glproc candidates that differ only in weight format, versus
   selecting between a glproc and a glcuda candidate for the same
   computation), this document SHALL separate them rather than treat them
   as one problem with one solution.

---

# Non Goals

- This document does not propose a solution or a combination method.
- This document does not evaluate GATE's constraint-validation step
  (`Validator`, `Constraint`); that step is assumed complete and correct
  per `architecture/GATE/GATE-algorithm.md`'s Theorem 10.1, and only the
  subsequent cost-evaluation step is this directory's concern.
- This document does not address distributed or multi-device selection;
  `architecture/GATE/GATE-mapping.md` already places that out of scope.

---

# Exit Criteria

Complete when the problem statement in Requirement 1 is specific enough
that a candidate solution can be judged to address it or not, without
ambiguity about what "solving" means.

---

# References

- ARTX01-Reality.md, ARTX02-Observation.md (evidentiary basis)
- ARTX04-Theory.md (what already exists against this problem statement)
- `architecture/GATE/GATE-algorithm.md` (where this problem sits in the
  overall GATE sequence)

---
---

# SOURCE: ARTX04-Theory.md

# Purpose

This document surveys GwenLand's Existing Diagnostic Models and, where
relevant, external prior art, so that no future cost model is designed
without first checking what already exists.

---

# Scope

Covers: `glbench`'s Roofline, Ceiling/Efficiency, and Bottleneck
Classifier components, and the external systems already surveyed in
`architecture/GATE/`'s own related-work material.

Excludes: judging where these components fall short of ARTX03's problem
statement (ARTX05).

---

# Inputs

- `glbench/src/analysis/roofline.rs`
- `glbench/src/analysis/ceiling.rs`
- `glbench/src/analysis/bottleneck.rs`
- `architecture/GATE/` sections referencing TVM, XLA, TensorRT, ONNX
  Runtime (already surveyed there; not re-surveyed here).

---

# Outputs

A named inventory of existing components, each with its stated purpose,
mechanism, and known limitation, for ARTX05 to compare against ARTX03.

---

# Requirements

1. This document SHALL record the Roofline component's purpose (classify
   one already-executed workload as bandwidth-bound or compute-bound by
   comparing arithmetic intensity to the Reference Tier's measured
   ceiling), its mechanism (a ratio of operation count to byte count,
   compared against a ratio of peak compute to peak bandwidth), and its
   known limitation (the ratio only reflects compute cost that is counted
   as an operation; per ARTX02, the component that actually exposed the
   two real throughput regressions on this project was a direct
   throughput measurement, not this ratio).
2. This document SHALL record the Ceiling/Efficiency component's purpose
   (produce a single ratio of observed throughput to a bandwidth ceiling,
   with the ceiling's provenance — measured on this machine, or read from
   a published specification — carried alongside the ratio), and its
   known limitation (a single dimension; requires an already-executed
   run; produces no answer when no ceiling is available, by design).
3. This document SHALL record the Bottleneck Classifier's purpose
   (produce one of a fixed set of categorical judgments from the Ceiling
   component's ratio, or decline to judge when insufficient signal
   exists), and its known limitation (categorical, not a ranking method;
   requires an already-executed run; considers only the ceiling ratio).
4. This document SHALL state plainly that none of the three components in
   Requirements 1 through 3 operate on more than one candidate at a time,
   and none operate before execution — both are structural facts about
   their design, not omissions to be read as failures.
5. This document MUST NOT propose modifications to any of these three
   components; recording their current behavior is this document's only
   responsibility.
6. External systems (TVM, XLA, TensorRT, ONNX Runtime) SHALL be referenced
   by pointing to their existing treatment in `architecture/GATE/`'s
   related-work material rather than re-described here, to avoid
   terminology or fact drift between the two directories.

---

# Non Goals

- This document does not evaluate glictus-caliburni's or any other
  crate's analysis capabilities beyond what `glbench` provides; no such
  survey has been performed.
- This document does not recommend reusing, extending, or replacing any
  surveyed component. That judgment, if warranted, belongs to ARTX05 and
  ultimately to a decision made outside this directory.

---

# Exit Criteria

Complete when every Existing Diagnostic Model bearing on ARTX03's problem
statement is named here with its purpose, mechanism, and limitation, each
traceable to a source file.

---

# References

- ARTX03-Problem.md (the problem these components are compared against)
- ARTX05-Gap.md (where the comparison is made explicit)
- `architecture/GATE/GATE-mapping.md` (glbench's ownership boundary within
  the GwenLand crate structure)

---
---

# SOURCE: ARTX05-Gap.md

# Purpose

This document states, explicitly and with evidence, exactly where the
components inventoried in ARTX04 fall short of the problem stated in
ARTX03. Every gap here is a specific, checkable claim, not a general
assessment.

---

# Scope

Covers: named gaps between Existing Diagnostic Models and the
candidate-ranking problem, each supported by an ARTX02 observation.

Excludes: assumptions a future solution would need to make (ARTX08) and
which quantities are involved (ARTX09) — this document identifies *that*
a gap exists, not how to close it.

---

# Inputs

- ARTX02-Observation.md, ARTX03-Problem.md, ARTX04-Theory.md.

---

# Outputs

A list of specific, evidenced gaps, each usable as a requirement input for
any future mathematical or architectural work.

---

# Requirements

1. Gap 1 SHALL be recorded as: every Existing Diagnostic Model surveyed
   in ARTX04 evaluates one already-executed run; none rank multiple
   not-yet-executed candidates. This is the entire subject of ARTX03's
   problem statement, and no existing component addresses it.
2. Gap 2 SHALL be recorded as: every existing measurement abstraction in
   the surveyed components (the Ceiling component's basis field, the
   Bottleneck Classifier's declination to judge, the energy-measurement
   component's absence-signaling behavior) treats "not measured" as a
   distinct, representable outcome, never defaulted to a number. A future
   cost model spanning multiple dimensions MUST preserve this property to
   be consistent with existing GwenLand practice; a representation that
   requires a numeric value for every dimension, always, does not.
3. Gap 3 SHALL be recorded as: per ARTX02's observation on throughput
   measurement, two known real regressions on this project were only
   exposed by direct throughput measurement, not by wall-clock time or
   memory footprint alone. Any future cost model limited to only time and
   memory as dimensions would repeat a blind spot this project has
   already paid to discover once.
4. Gap 4 SHALL be recorded as: the Roofline component's own design assumes
   workload behavior can be classified from structural counts
   (operations, bytes) without executing the workload. Per ARTX02's
   observation on the native low-bit-width format, a structural estimate
   of this kind was wrong, by a large margin, for exactly the class of
   decision (weight-format selection) a future cost model would need to
   make. This is recorded as a gap in *analytical estimation as a
   method*, not a gap in the Roofline component specifically.
5. Every gap recorded here MUST cite the specific ARTX02 observation(s)
   that support it. A gap without a supporting observation MUST NOT be
   recorded; it belongs in ARTX11 as an item requiring further
   observation first.

---

# Non Goals

- This document does not propose how to close any gap; that is
  downstream work for a future mathematical specification, informed by
  ARTX08 through ARTX12.
- This document does not claim these four gaps are exhaustive. Only gaps
  supported by an existing ARTX02 observation are recorded; absence of a
  fifth gap here reflects absence of supporting observation, not absence
  of risk.

---

# Exit Criteria

Complete when each gap is stated as a specific, falsifiable claim with a
named supporting observation, such that a reader could check the claim
against ARTX02 without taking this document's word for it.

---

# References

- ARTX02-Observation.md (evidence for each gap)
- ARTX03-Problem.md (the problem each gap is measured against)
- ARTX04-Theory.md (the components each gap concerns)
- ARTX08-Assumptions.md (assumptions these gaps place under scrutiny)

---
---

# SOURCE: ARTX06-Terminology.md

# Purpose

This document is the single source of truth for every term used in this
directory. No other document in this directory may define or redefine a
term; all of them reference this one.

---

# Scope

Covers: definitions for terms coined by this directory, and pointers to
terms already canonical in `architecture/GATE/` that this directory reuses
without change.

Excludes: any term specific to a single ARTX document that is not reused
elsewhere — such a term, if truly local, is defined inline in that
document instead of listed here.

---

# Inputs

- `architecture/GATE/GATE-concepts.md` (source of every reused term).
- ARTX01 through ARTX05 (source of every term coined by this directory).

---

# Outputs

A closed glossary every other document in this directory, and any future
document referencing this directory, MUST use without variation.

---

# Requirements

1. Terms reused from `architecture/GATE/` without redefinition:
   - **ExecutionPlan**, **Constraint**, **Validator**, **MetricVector**,
     **WeightVector**, **ExecutionPolicy**, **BackendKind** — defined in
     `architecture/GATE/GATE-concepts.md`. This directory MUST use these
     names exactly as defined there and MUST NOT introduce a synonym for
     any of them.
2. Terms coined by this directory, defined here and only here:
   - **Reference Tier**: the specific hardware and host-operating-system
     configuration this directory's observations are grounded in (an
     Intel Tiger Lake-class CPU, dual-channel DDR4 memory, and a Windows
     host, per ARTX02). A claim not qualified as tier-specific SHALL NOT
     be assumed to hold on a different Reference Tier.
   - **Analytical Estimate**: a value for a `MetricVector` field produced
     from `ExecutionPlan` structure alone, without executing the plan.
   - **Calibrated Measurement**: a value for a `MetricVector` field
     produced by executing the `ExecutionPlan`, or a representative proxy
     of it, at least once.
   - **Undetermined State**: the explicit, representable absence of a
     value for a `MetricVector` field, distinct from any numeric default.
   - **Backend-Conditional Dimension**: a `MetricVector` field whose
     applicability or measurability depends on which `BackendKind` an
     `ExecutionPlan` targets.
   - **Existing Diagnostic Model**: any already-implemented GwenLand
     component (per ARTX04: Roofline, Ceiling/Efficiency, Bottleneck
     Classifier) that evaluates a single already-executed run.
   - **Decision Model**: the not-yet-built architectural component
     responsible for ranking multiple `ExecutionPlan` candidates before
     any of them executes — the subject this entire directory concerns.
   - **Combination Rule**: the unspecified procedure by which multiple
     `MetricVector` fields would be reduced to a single ranking decision.
     This term names an open question; this directory MUST NOT resolve it
     with a specific rule, per ARTX00's mathematics prohibition.
   - **Cross-Backend Comparison**: the act of ranking `ExecutionPlan`
     candidates that target different `BackendKind` values against one
     another, as opposed to ranking candidates that all target the same
     `BackendKind`.
3. Every term listed in Requirement 2 MUST be used identically, in this
   exact form, in every other document of this directory. A document
   using a synonym (for example, "target hardware" in place of
   `BackendKind`, or "un-executed alternative" in place of
   `ExecutionPlan` candidate) SHALL be corrected before this directory is
   considered consistent.
4. A new term MUST NOT be introduced anywhere in this directory without
   being added to this document in the same change.

---

# Non Goals

- This document does not define terms belonging to `architecture/GATE/`
  itself; it only points to them.
- This document does not define implementation-level names (Rust type or
  function names) beyond those already established in
  `architecture/GATE/GATE-concepts.md`; this directory is architecture,
  not implementation, per ARTX00.

---

# Exit Criteria

Complete when every term used more than once across ARTX01–ARTX05 and
ARTX07–ARTX12 appears in this document exactly once, with no synonym in
use anywhere else in the directory.

---

# References

- `architecture/GATE/GATE-concepts.md` (authoritative source for reused
  terms)
- ARTX07-Ontology.md (the structural relationships between these terms)
- Every other document in this directory (consumers of this glossary)

---
---

# SOURCE: ARTX07-Ontology.md

# Purpose

This document defines the structural relationships between the terms
registered in ARTX06 — which entities are composed of, conditioned by, or
produced from which others. It is a relational map, not a glossary and
not a causal analysis.

---

# Scope

Covers: is-composed-of, is-conditioned-by, and produces relationships
between ARTX06's registered terms.

Excludes: term definitions (ARTX06), empirically observed causal or
correlational relationships between measured quantities (ARTX10), and any
claim about which relationships hold on evidence rather than by
construction.

---

# Inputs

- ARTX06-Terminology.md.
- `architecture/GATE/GATE-concepts.md` (structure of `ExecutionPlan` and
  `MetricVector` this document's relationships must not contradict).

---

# Outputs

A structural map later documents (ARTX08, ARTX09, ARTX10) can reference
by relationship name rather than re-deriving.

---

# Requirements

1. An `ExecutionPlan` SHALL be recorded as targeting exactly one
   `BackendKind`; a `Decision Model` compares `ExecutionPlan` instances
   that MAY target the same `BackendKind` (a same-backend comparison) or
   different ones (a `Cross-Backend Comparison`), and this directory MUST
   treat these as structurally distinct cases wherever a relationship is
   stated to depend on which one applies.
2. An `ExecutionPlan`'s `MetricVector` SHALL be recorded as composed of
   individually named fields (per `architecture/GATE/GATE-concepts.md`);
   each field MAY independently be a `Backend-Conditional Dimension`, and
   this document MUST NOT assume a field's conditionality transfers to
   any other field.
3. A `MetricVector` field's value SHALL be recorded as produced by
   exactly one of: an `Analytical Estimate`, a `Calibrated Measurement`,
   or an `Undetermined State` — these three MUST be treated as mutually
   exclusive outcomes for a given field on a given `ExecutionPlan`, never
   as a spectrum or a default-to-one-another relationship.
4. A `Decision Model` SHALL be recorded as consuming a set of
   `ExecutionPlan` candidates (already filtered by a `Validator` per
   `architecture/GATE/GATE-algorithm.md`) and producing a ranking or
   selection among them via an unspecified `Combination Rule`; this
   document MUST NOT specify what the `Combination Rule` is, only that a
   `Decision Model` requires one.
5. An `Existing Diagnostic Model` SHALL be recorded as consuming exactly
   one already-executed run and producing an interpretation of it (a
   ratio, or a categorical judgment); it MUST NOT be recorded as consuming
   multiple candidates or as producing a ranking — that distinguishes it
   structurally from a `Decision Model`, per ARTX04 and ARTX05's Gap 1.
6. A `Reference Tier` SHALL be recorded as a property of the observation,
   not of the `ExecutionPlan` or `BackendKind` — the same `ExecutionPlan`
   description could in principle be evaluated against a different
   `Reference Tier`, and this document MUST NOT conflate a tier-specific
   finding with a property of the plan itself.

---

# Non Goals

- This document does not assert that any of these relationships have been
  empirically confirmed; it only asserts that they hold by definition,
  given ARTX06's terms. Empirical confirmation is ARTX10 and ARTX11's
  responsibility.
- This document does not model relationships involving `Constraint` or
  `Validator` beyond Requirement 4's minimal reference; their internal
  structure is `architecture/GATE/`'s concern, not this directory's.

---

# Exit Criteria

Complete when every term in ARTX06 that participates in a structural
relationship with another term has that relationship stated exactly once
in this document.

---

# References

- ARTX06-Terminology.md (the terms related here)
- ARTX09-Variables.md, ARTX10-Dependencies.md (build on this structure
  with concrete quantities and empirical relationships)
- `architecture/GATE/GATE-concepts.md` (the `ExecutionPlan`/`MetricVector`
  structure this document's relationships extend)

---
---

# SOURCE: ARTX08-Assumptions.md

# Purpose

This document lists every assumption a future Decision Model would depend
on, states why each is necessary, whether it is experimentally
verifiable, and under what condition it fails. No assumption here is
endorsed; each is recorded for scrutiny.

---

# Scope

Covers: assumptions implied by the cost-model design surveyed in
`architecture/GATE/GATE-concepts.md`, evaluated against ARTX01–ARTX05.

Excludes: which quantities are involved (ARTX09) and how each assumption
would be checked (ARTX11) — this document only states that an assumption
exists and whether it currently holds.

---

# Inputs

- `architecture/GATE/GATE-concepts.md` (the drafted cost-model design
  these assumptions come from).
- ARTX02, ARTX05 (evidence for failure conditions).

---

# Outputs

A list of assumptions, each an explicit input to ARTX11 (how to validate)
and ARTX12 (whether readiness can be declared while it stands
unresolved).

---

# Requirements

Each assumption below SHALL be recorded with: why it is necessary, whether
it is experimentally verifiable, and the condition under which it fails.

1. **All `MetricVector` fields are comparably meaningful across every
   `BackendKind`.** Necessary for a uniform `Cross-Backend Comparison`.
   Verifiable by inspecting whether a given `BackendKind`'s pipeline (per
   ARTX01) has a physical referent for the field in question. Fails for
   at least one field (a synchronization-related dimension) on the CPU
   `BackendKind`, which has no launch or stream concept at all — this is
   not a low measured value, it is the absence of a referent.
2. **`MetricVector` fields can be produced as an `Analytical Estimate`
   without executing the `ExecutionPlan`.** Necessary for the drafted
   design's stated goal of adding no execution overhead to cost
   evaluation. Verifiable by comparing an `Analytical Estimate` against a
   `Calibrated Measurement` for the same candidates. Already found to fail
   for at least one field (a timing-related dimension) in the specific
   case of weight-format candidate selection, per ARTX02's observation on
   the native low-bit-width format.
3. **Multiple `MetricVector` fields can be combined into a single ranking
   by a `Combination Rule` that treats each field as substitutable for
   the others at some fixed rate.** Necessary for a `Decision Model` to
   produce a total order over candidates. Not verifiable without a
   defined `Combination Rule`, which this directory does not specify.
   Known to be a risky assumption: ARTX02's observation on the native
   low-bit-width format describes a case where a resource saving on one
   dimension should not have offset a wall-clock loss on another, at any
   substitution rate, for the candidate to remain acceptable.
4. **A missing value for a `MetricVector` field can be safely replaced
   with a default value for the purpose of computing a ranking.** Not
   stated explicitly by the drafted design, but required by a
   representation that has no `Undetermined State`. Verifiable: fails
   whenever a field is genuinely unmeasurable rather than merely
   unmeasured this run — per ARTX02, this holds unconditionally for the
   energy-related dimension on the Reference Tier's host operating
   system.
5. **Preset `WeightVector` values reflect real deployment tradeoffs.**
   Necessary for `ExecutionPolicy` presets to be meaningful rather than
   arbitrary. Not verifiable at present: no sensitivity analysis
   connecting these values to a `Reference Tier` observation exists in
   any source this directory has reviewed. Not yet falsified, only
   because nothing has been measured against them.

---

# Non Goals

- This document does not resolve any assumption; resolution requires
  either a validation method (ARTX11) or a decision made outside this
  directory.
- This document does not assume these five are exhaustive; an assumption
  without a stated necessity and failure condition MUST NOT be added
  here without both.

---

# Exit Criteria

Complete when every assumption identifiable from
`architecture/GATE/GATE-concepts.md`'s cost-model design has a stated
necessity, verifiability, and failure condition, each traceable to an
ARTX01, ARTX02, or ARTX05 finding where one is claimed to already fail.

---

# References

- `architecture/GATE/GATE-concepts.md` (source of the assumptions)
- ARTX05-Gap.md (evidence several assumptions already fail)
- ARTX11-Validation.md (how remaining assumptions would be checked)
- ARTX12-Ready.md (what unresolved assumptions mean for readiness)

---
---

# SOURCE: ARTX09-Variables.md

# Purpose

This document records every candidate measurable quantity relevant to a
future Decision Model, each with its physical meaning, unit, measurement
method, and whether it is independent or derived from another quantity.
No quantity is combined with another here; combination is out of scope
for this entire directory.

---

# Scope

Covers: the five `MetricVector` fields already named in
`architecture/GATE/GATE-concepts.md`, plus one additional quantity this
directory's evidence identifies as missing from that set.

Excludes: any relationship between these quantities (ARTX10) and how each
would be validated (ARTX11).

---

# Inputs

- `architecture/GATE/GATE-concepts.md` (the five named fields).
- ARTX01, ARTX02 (physical grounding and measurement evidence).

---

# Outputs

A table of quantities, each with enough information for ARTX10 to state
relationships between them without re-deriving their definitions.

---

# Requirements

Each quantity SHALL be recorded with: physical meaning, measurement method
today (if any), and whether it is independent or derived from another
quantity.

1. **Latency** (`MetricVector` field `latency_ms`). Physical meaning:
   wall-clock execution time. Measured today via elapsed-time
   instrumentation already present in glproc and glcuda. Derived, not
   independent: per ARTX02 and ARTX10, its value is downstream of either
   achieved memory-read throughput or achieved compute throughput,
   depending on which regime a given stage falls into — a fact
   established per-stage, not fixed in advance.
2. **Peak memory** (`MetricVector` field `peak_memory_mb`). Physical
   meaning: maximum live allocation. Measured today via the existing
   memory-telemetry component. Independent of the other quantities listed
   here as a direct measurement, though ARTX10 records an empirical
   relationship between it and Latency.
3. **Synchronization overhead** (`MetricVector` field
   `sync_overhead_ms`). Physical meaning: time spent on kernel launch,
   transfer, and barrier operations. Measurable on the glcuda
   `BackendKind` (which has a stream-synchronization concept); has no
   physical referent on the glproc `BackendKind`, per ARTX01. A
   Backend-Conditional Dimension, per ARTX06 and ARTX07.
4. **Energy** (`MetricVector` field `energy_mj`). Physical meaning:
   package energy consumption. Measurable only through a Linux-specific
   kernel interface; unconditionally in an Undetermined State on the
   Reference Tier's host operating system, per ARTX02. No proxy for this
   quantity exists anywhere in the reviewed sources.
5. **Numerical deviation** (`MetricVector` field `numerical_error`, as
   specified in `architecture/GATE/GATE-concepts.md`: a continuous
   relative deviation from a reference output). No instrumentation
   producing this specific quantity exists in the reviewed sources; it is
   recorded as `UNKNOWN` whether it is measurable today.
6. **Discrete token agreement** (not currently a named `MetricVector`
   field). Physical meaning: the count of leading output tokens matching
   a designated reference engine's output under fixed-seed, non-sampling
   decoding. Measured today by an existing comparison component, per
   ARTX02. This document records it as a *distinct* quantity from
   Requirement 5's numerical deviation, not as a measurement of it — the
   two can diverge in either direction, per ARTX02's determinism
   qualification, and this directory MUST NOT treat one as a proxy for
   the other without a stated justification, which none of the reviewed
   sources provide.
7. **Achieved compute throughput relative to achieved memory throughput**
   (not currently a named `MetricVector` field). Physical meaning: a
   measure of whether a stage's execution time is limited by
   memory-bus transfer or by arithmetic execution. Measured today via
   existing per-stage throughput instrumentation. This document records
   it as the quantity that, per ARTX05's Gap 3, already exposed two real
   regressions that Requirements 1 and 2 alone did not — and as such, a
   candidate quantity any future Decision Model's variable set should
   consider, without this document prescribing that it must be included.

---

# Non Goals

- This document does not assign a unit conversion or normalization
  procedure between quantities — that is mathematical work, out of scope
  per ARTX00.
- This document does not state an expected numeric range for any
  quantity beyond what ARTX02 already records; inventing a range without
  a measurement would violate ARTX00's prohibition on fabricated
  evidence.

---

# Exit Criteria

Complete when every `MetricVector` field named in
`architecture/GATE/GATE-concepts.md` has an entry here stating whether it
is measurable on the Reference Tier today, and when every quantity
identified in ARTX05 as missing from that field set is recorded as a
candidate.

---

# References

- `architecture/GATE/GATE-concepts.md` (the five named fields)
- ARTX02-Observation.md (measurement evidence)
- ARTX05-Gap.md (motivation for Requirement 7)
- ARTX10-Dependencies.md (relationships between these quantities)

---
---

# SOURCE: ARTX10-Dependencies.md

# Purpose

This document records causal and conditional relationships observed
between the quantities named in ARTX09 — which quantity influences
another, which are independent, and which cannot exist simultaneously. It
records observed relationships, not correlations assumed by convenience.

---

# Scope

Covers: relationships supported by an ARTX02 observation or an ARTX07
structural rule.

Excludes: how to combine these quantities into a ranking (mathematical
work, out of scope per ARTX00) and how to validate a relationship not yet
supported by evidence (ARTX11).

---

# Inputs

- ARTX02-Observation.md, ARTX07-Ontology.md, ARTX09-Variables.md.

---

# Outputs

A dependency record that ARTX11 uses to decide what still needs checking,
and that ARTX12 uses to judge whether the variable set in ARTX09 is
well-understood enough to hand to mathematical work.

---

# Requirements

1. **Latency is downstream of Peak Memory's related transfer volume, or
   of Achieved Compute Throughput, depending on regime.** This document
   SHALL record that which of the two dominates a given stage's Latency
   is itself an empirical, per-stage fact (established by the Existing
   Diagnostic Model surveyed in ARTX04), not a fixed relationship that
   holds the same way for every `ExecutionPlan`.
2. **Peak Memory and Latency are not freely tradeable in the direction
   commonly assumed.** This document SHALL record the specific,
   already-observed counter-example from ARTX02: reducing a candidate's
   memory footprint (a smaller weight format) *increased* its Latency,
   contrary to an assumption that lower memory footprint implies lower or
   equal latency. Any future Decision Model design that treats these two
   quantities as independent, separately-weighted axes MUST account for
   this observed counter-example rather than assume it away.
3. **Synchronization Overhead is conditioned on `BackendKind`.** This
   document SHALL record that this quantity is only meaningful when the
   `ExecutionPlan` targets the glcuda `BackendKind`, per ARTX01 and
   ARTX09; it MUST NOT be treated as a small-but-present value on the
   glproc `BackendKind` — it has no referent there at all, per ARTX09's
   Backend-Conditional Dimension classification.
4. **Energy is conditioned on host-operating-system capability, a
   variable entirely external to the `ExecutionPlan` being evaluated.**
   This document SHALL record that this condition does not vary between
   candidates within a single `Decision Model` invocation on the
   Reference Tier — it is either available for none of them or,
   per ARTX02, unavailable for all of them.
5. **Discrete Token Agreement is conditioned on decoding policy.** This
   document SHALL record that this quantity is only well-defined under
   fixed-seed, non-sampling decoding, per ARTX02's determinism
   qualification; under a sampling decoding policy, this document records
   its well-definedness as `UNKNOWN`.
6. This document MUST NOT assert a dependency that is not supported by an
   ARTX02 observation or an ARTX07 structural rule. A suspected but
   unconfirmed dependency belongs in ARTX11 as an item to validate, not
   here as a recorded fact.

---

# Non Goals

- This document does not quantify the strength of any relationship (no
  ratios, no coefficients); it records direction and conditionality only.
- This document does not propose a variable set revision; ARTX09 already
  records the candidate quantities, and any revision to that set is
  downstream work outside this directory.

---

# Exit Criteria

Complete when every quantity in ARTX09 has its known conditioning
factors and known relationships to other quantities recorded, each
traceable to a specific ARTX02 observation or ARTX07 rule.

---

# References

- ARTX02-Observation.md, ARTX07-Ontology.md, ARTX09-Variables.md
- ARTX11-Validation.md (unconfirmed relationships requiring validation)
- ARTX12-Ready.md (what these dependencies mean for readiness)

---
---

# SOURCE: ARTX11-Validation.md

# Purpose

This document defines what method would be required to confirm or refute
each unresolved assumption from ARTX08 and each unconfirmed relationship
from ARTX10. It defines the validation method; it does not perform the
validation.

---

# Scope

Covers: a named validation method for each open item carried from ARTX08
and ARTX10.

Excludes: performing any measurement, running any experiment, or
resolving any assumption. This document's output is a set of pending
action items, not a set of answers.

---

# Inputs

- ARTX08-Assumptions.md, ARTX10-Dependencies.md.
- `gl-agent-skills/bench-skills/measurement-discipline.md` (the project's
  existing standard for what counts as a credible measurement).

---

# Outputs

A list of pending validation action items, each with a named method,
usable as an input to future work and to ARTX12's readiness judgment.

---

# Requirements

1. For Assumption 1 (ARTX08: cross-`BackendKind` comparability of every
   `MetricVector` field): the validation method SHALL be static
   inspection of each `BackendKind`'s implementation for a physical
   referent of each field, repeated whenever a new `BackendKind` is added
   to the candidate set under consideration. This is already partially
   complete for glproc and glcuda per ARTX01 and MUST be repeated for
   glvulkan and glmetal once either has a non-stub implementation.
2. For Assumption 2 (ARTX08: analytical estimation without execution):
   the validation method SHALL be a production comparison, on the
   Reference Tier, between an `Analytical Estimate` and a `Calibrated
   Measurement` for the same set of `ExecutionPlan` candidates, following
   the same production A/B discipline already established in
   `gl-agent-skills/bench-skills/measurement-discipline.md`. A probe or
   microbenchmark comparison MUST NOT substitute for this method,
   consistent with existing project practice recorded in
   `gl-agent-skills/cpu-skills/rejected-optimizations.md`.
3. For Assumption 3 (ARTX08: substitutability of `MetricVector` fields
   under a `Combination Rule`): the validation method SHALL be deferred
   to whichever future mathematical specification defines a `Combination
   Rule`, since no such rule exists yet to test. This document records
   the requirement that any such rule, once proposed, MUST be checked
   against the counter-example recorded in ARTX10 before being adopted.
4. For Assumption 4 (ARTX08: safety of defaulting a missing value): the
   validation method SHALL be a review of every `Backend-Conditional
   Dimension` and every host-capability-conditioned quantity identified
   in ARTX09 and ARTX10, confirming each has an explicit `Undetermined
   State` representation before any `Combination Rule` is implemented
   against it.
5. For Assumption 5 (ARTX08: preset `WeightVector` values reflecting real
   tradeoffs): the validation method SHALL be a sensitivity study —
   varying weight values against a fixed, real candidate set and Reference
   Tier and observing whether the resulting selection changes align with
   the deployment context each preset claims to serve. This study MUST
   NOT be performed until Assumption 2's validation is complete, since a
   sensitivity study over unreliable `Analytical Estimate` values would
   produce an uninterpretable result.
6. For each dependency recorded as `UNKNOWN` in ARTX09 or ARTX10
   (Numerical Deviation's measurability; Discrete Token Agreement's
   well-definedness under sampling), the validation method SHALL be: build
   the missing instrumentation first, then re-observe, following the same
   evidence standard ARTX02 already applies. This document MUST NOT
   propose a shortcut that infers an `UNKNOWN` value without new
   instrumentation.

---

# Non Goals

- This document does not assign an owner, a deadline, or a priority order
  to these validation items; that is project-management work, out of
  scope for an architecture specification.
- This document does not validate anything itself. An item listed here
  remains open until a dated, sourced observation is added to a revision
  of ARTX02 or ARTX08.

---

# Exit Criteria

Complete when every unresolved assumption in ARTX08 and every unconfirmed
or `UNKNOWN` relationship in ARTX09/ARTX10 has exactly one named
validation method recorded against it.

---

# References

- ARTX08-Assumptions.md, ARTX09-Variables.md, ARTX10-Dependencies.md
  (the open items this document addresses)
- `gl-agent-skills/bench-skills/measurement-discipline.md`,
  `gl-agent-skills/cpu-skills/rejected-optimizations.md` (the project's
  existing evidentiary standard this document adopts rather than
  inventing a new one)
- ARTX12-Ready.md (what remains open after this document is applied)

---
---

# SOURCE: ARTX12-Ready.md

# Purpose

This document states whether enough architectural understanding exists,
per ARTX00 through ARTX11, to proceed to a separate mathematical
specification for a GATE Decision Model. It is the final judgment this
directory produces.

---

# Scope

Covers: a readiness verdict, the specific items outstanding before
mathematical work should begin, and where that mathematical work belongs.

Excludes: performing that mathematical work. No equation, symbol, or
objective function appears in this document, consistent with every
other document in this directory.

---

# Inputs

- ARTX05-Gap.md, ARTX08-Assumptions.md, ARTX10-Dependencies.md,
  ARTX11-Validation.md.

---

# Outputs

A verdict of READY, PARTIALLY READY, or NOT READY, and a list of
outstanding action items gating any upgrade of that verdict.

---

# Requirements

1. The verdict SHALL be recorded as **PARTIALLY READY**. The problem is
   well-defined (ARTX03), the gap against existing components is
   evidenced (ARTX05), and the candidate variable set is identified with
   known conditioning factors (ARTX09, ARTX10). Mathematical work MUST
   NOT begin, however, until the outstanding items in Requirement 2 are
   closed, because at least one of them (Assumption 2, ARTX08) is not a
   theoretical risk but an already-observed failure mode for the specific
   class of decision this Decision Model exists to make.
2. The following items SHALL be closed before this verdict may be
   upgraded to READY:
   - A working set of more than one real `ExecutionPlan` candidate MUST
     exist to observe `MetricVector` values against. None exists in the
     reviewed sources at the time of this document.
   - The production comparison defined in ARTX11 (Analytical Estimate
     versus Calibrated Measurement) MUST be performed against that
     candidate set before any `Combination Rule` is designed assuming
     analytical estimation is sufficient.
   - An explicit decision MUST be made on whether Synchronization
     Overhead and Energy are members of a universal, cross-backend
     `MetricVector`, or are instead backend-conditional annotations
     outside a `Cross-Backend Comparison`'s scope — this document
     recommends the latter, per ARTX09 and ARTX10, but does not decide it
     unilaterally.
   - An explicit decision MUST be made on whether Numerical Deviation, as
     specified in `architecture/GATE/GATE-concepts.md`, is built as new
     instrumentation or is redefined to mean the already-measured
     Discrete Token Agreement quantity — these are not interchangeable
     per ARTX09, and the specification currently names the former while
     only the latter exists.
3. Mathematical work, once these items are closed, SHALL be conducted in
   a separate specification outside `architecture/`, per this megaprompt's
   own governing rule that architecture does not contain mathematics —
   a `math/` or `theory/` directory is the appropriate location, not a
   further ARTX document here and not a further revision of
   `architecture/GATE/GATE-concepts.md` in place.
4. This document MUST NOT be revised to READY on the basis of confidence
   alone. An upgrade requires a dated, sourced update to ARTX02 or ARTX08
   closing one of Requirement 2's items, following the validation methods
   ARTX11 already defines.
5. This document SHALL note, for traceability, that a Combination Rule
   favoring Pareto-style filtering over unconditional linear weighting
   was identified in the underlying research as better aligned with the
   evidence in ARTX10 (a resource saving on one dimension should not
   universally offset a loss on another). This document records that
   observation as an input to future mathematical work and explicitly
   does not adopt it as an architectural decision, since selecting a
   Combination Rule is mathematical, not architectural, work.

---

# Non Goals

- This document does not set a timeline for closing the outstanding
  items; that belongs to whatever wave-planning document (for example,
  `architecture/GATE/GATE-impl-plan.md`) governs the implementation
  effort this research feeds into.
- This document does not authorize any specific engineer or change to
  close the outstanding items; it only states that they are outstanding.

---

# Exit Criteria

This document, and this directory, are complete when this verdict and
its outstanding items are stated clearly enough that a future
mathematical specification can begin by reading this document alone and
knowing exactly what must be true before it starts.

---

# References

- ARTX05-Gap.md, ARTX08-Assumptions.md, ARTX10-Dependencies.md,
  ARTX11-Validation.md (the basis for this verdict)
- `architecture/GATE/GATE-concepts.md` (the specification this verdict
  gates further revision of)
- `architecture/GATE/GATE-impl-plan.md` (the wave plan this verdict is an
  input to, specifically its Wave G3 scope)

---
---

END OF FULL READ.
