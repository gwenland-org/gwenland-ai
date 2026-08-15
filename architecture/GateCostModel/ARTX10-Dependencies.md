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
