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
