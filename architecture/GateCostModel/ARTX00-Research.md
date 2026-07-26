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
