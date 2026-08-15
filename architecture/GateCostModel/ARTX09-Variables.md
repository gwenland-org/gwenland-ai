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
