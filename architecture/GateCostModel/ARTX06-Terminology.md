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
