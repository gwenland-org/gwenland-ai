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
