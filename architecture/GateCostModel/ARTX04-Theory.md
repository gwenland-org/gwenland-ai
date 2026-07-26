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
