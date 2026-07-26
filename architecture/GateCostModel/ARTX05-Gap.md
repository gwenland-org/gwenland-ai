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
