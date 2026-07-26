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
