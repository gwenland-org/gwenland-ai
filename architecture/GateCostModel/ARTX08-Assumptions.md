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
