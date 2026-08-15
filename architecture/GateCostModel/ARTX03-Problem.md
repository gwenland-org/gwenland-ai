# ARTX03 — Problem Statement
## GateCostModel · GwenLand AI
## Last updated: 2026-07-25

---

## Purpose

This document states the specific engineering problem a GATE cost model
would need to solve, rejecting vague framings ("the system is slow," "the
cost model must be accurate") in favor of a precise statement traceable to
ARTX01-Verified.md and ARTX02-Observation.md. It exists so ARTX04 (what
already exists against this problem) and ARTX05 (where existing solutions
fall short) have a fixed, unambiguous target to be evaluated against.

---

## The Problem, Stated Precisely

**Given more than one `ExecutionPlan` candidate for the same computation,
each already known to satisfy every registered `Constraint`, determine
which candidate to dispatch — without executing every candidate to find
out.**

This is Step 5 of GATE's seven-step sequence (`architecture/GATE/
GATE-algorithm.md`, "Evaluate cost"): by the time this problem is reached,
Steps 1–4 (generate, validate, reject invalid, error on empty valid set)
have already run. The input to this problem is `𝒫valid` — a finite,
non-empty set of plans, each already proven correct by `Validator::
validate` (Theorem 10.1: `V(P) = 1 ⟹ P` satisfies all `k` registered
constraints). The problem this document names is exactly: `𝒞(P, w) =
wᵀ·m(P)` for every `P ∈ 𝒫valid`, then `P* = argmin 𝒞(P, w)` — and,
specifically, how `m(P)` (the plan's 5-dimensional metric vector) is
obtained without running `P` to observe it.

Two things must be true of `m(P)` for this problem to be solved rather
than merely restated:
- It must be obtainable **before** dispatch (a metric read off of the
  *executed* plan is not a cost model, it is a benchmark result).
- It must be **accurate enough**, relative to the policy's weight vector
  `w`, that `argmin 𝒞(P, w)` over the *estimated* metrics agrees with what
  `argmin` over the *true, measured* metrics would have selected — at
  least often enough to be worth the estimate's cost.

---

## What This Problem Is Not

Per ARTX01's own distinction (Requirement 4: physical operations vs.
software-only abstractions) and Non Goal 2 ("this document does not judge
whether either pipeline is efficient or well-architected"), this document
rejects two adjacent but different framings:

1. **This is not "make glproc/glcuda faster."** Raw engine throughput —
   decode tok/s, prefill tok/s, GB/s against the measured ceiling — is the
   subject of ARTX02's OBS-07 through OBS-09 and of separate,
   non-GateCostModel performance work (Veritas-series sprints). Improving
   an engine's actual execution speed changes `m(P)` for a *given* `P`; it
   does not touch the decision problem of *choosing among* multiple `P`
   without running them. A cost model that perfectly estimates `m(P)` for
   a slow engine has still solved this document's problem; it has not
   made the engine faster.

2. **This is not "the cost model must be accurate."** Restated this way,
   the problem is untestable — there is no threshold at which an estimate
   becomes "accurate" in the abstract. The testable version is the
   agreement condition above: does `argmin` over estimated `m(P)` select
   the same plan `argmin` over measured `m(P)` would have selected, under
   the weight vector `w` a given policy actually uses. This is a
   decision-agreement question, not a numerical-precision question, and
   the difference matters because a coarse estimate can still get the
   *decision* right if the candidates are far enough apart in true cost —
   this document's problem is about the decision, not the estimate's
   precision for its own sake.

---

## This Is a Decision Problem, Not a Resource-Bound Problem

Per ARTX01 Requirement 4 and the `architecture/GATE/GATE-algorithm.md`
empirical finding (validation plus cost-minimal selection measured at
**1.4–8.2 µs per policy** for `n=6` candidates, seven orders of magnitude
below a single inference at 376 ms): the decision itself consumes no
measurable hardware resource of its own.

- `ExecutionPlan` (`glcore/src/gate/plan.rs:72-81`) is a struct: an
  ordering, a `BackendKind` enum, a `HashMap<OpId, MemoryLayout>`, and a
  `MetricVector`. Holding one, cloning one, or comparing one to another
  touches no DRAM bandwidth budget, no PCIe link, no vector unit.
- `Constraint::validate` (`glcore/src/gate/constraint.rs:10-16`) is a pure
  function from a plan to `{Pass, Reject}`. Running seven of them, per the
  measured `O(knp)` bound, costs microseconds — not because the
  implementation happens to be fast today, but because nothing in the
  operation's definition requires touching a physical resource at all.
- `ExecutionPolicy::weight_vector` (`glcore/src/gate/policy.rs:23-38`) is a
  table lookup into a fixed `[f64; 5]` array.

The **resource-bound part of the overall system** is real and is where
ARTX01/ARTX02's physical facts belong — DRAM bandwidth is consumed when a
selected plan is *dispatched* and *runs* on glproc (Step 7), not when it
is *evaluated* for selection (Step 5). Conflating these two would make
every cost-model claim in this directory unfalsifiable: a slow decision
step and a slow dispatched engine would look identical from the outside,
and only one of them is this document's problem.

---

## The Three Physical Subsystems the Decision Must Account For

Per ARTX01 Requirement 4 (as verified in ARTX01-Verified.md's R1–R3), the
decision problem's `m(P)` estimates must have a **physical referent** in
one of three subsystems, each backend-specific and not interchangeable:

1. **DRAM bandwidth (glproc decode).** Confirmed bandwidth-adjacent,
   serial, no cross-token reuse (ARTX01-Verified R1, R5; ARTX02 OBS-01,
   OBS-04, OBS-07). A candidate plan targeting `BackendKind::Glproc` for a
   decode-phase operation has its cost dominated by bytes streamed once
   per token against this ceiling.
2. **Vector-unit throughput for weight-unpacking arithmetic (glproc
   decode, backend-specific).** Confirmed as a *separate* bottleneck from
   (1) by ARTX02 OBS-02/OBS-03: native Q4_K measured lower GMAC/s than
   Q8_0 with the gap persisting when the working set was L2-resident,
   proving the bound is compute (nibble-unpack arithmetic), not bandwidth,
   for that specific weight format. A cost model that scores every glproc
   candidate purely on bytes-moved (subsystem 1) would have mispredicted
   this exact case — the format-dependent unpack cost is a second,
   independent physical referent, not a refinement of the first.
3. **Host-device transfer plus stream synchronization (glcuda).**
   Confirmed as PCIe transfer for per-token embedding upload and logits
   download, plus one `cuCtxSynchronize` per token (ARTX01-Verified R2,
   R3) — a substrate with no glproc analogue, since glproc has no
   host/device boundary to cross.

**These three subsystems are not comparable to each other without further
support.** ARTX02 records no bandwidth-per-dollar, latency-per-byte, or
other common unit that would let a plan's DRAM-bandwidth cost on glproc be
placed on the same numeric scale as a plan's PCIe-transfer cost on glcuda
by any means this directory has verified. Where a cost model needs to
compare a glproc candidate against a glcuda candidate for the *same*
computation, the unit-conversion question is open and is **not resolved
by this document** — it is deferred to wherever ARTX09 (Variables) defines
the metric vector's actual units, and flagged here so ARTX04/ARTX05 do not
assume it is already solved.

---

## Distinct Problems Within This Problem

Per this document's own Requirement 5, at least two structurally different
selection problems exist under the one statement above, and a solution to
one does not imply a solution to the other:

### Problem A — Same-backend candidate selection

Selecting between two `glproc` candidates that differ only in weight
format (for example: a Q4_K-native candidate vs. a Q8_0-repacked
candidate for the same tensor). Both candidates share subsystem (1) and
potentially differ only in subsystem (2) — the vector-unit throughput for
their respective unpack arithmetic. ARTX02 OBS-02/OBS-03 is a **worked,
measured instance** of exactly this problem: the native-Q4_K candidate was
generated, and *would have been* selected by any cost model scoring purely
on bytes-per-token (subsystem 1 alone), since it moves fewer bytes — the
measured -33% end-to-end outcome is what a cost model failing to account
for subsystem (2) as a distinct term would produce.

### Problem B — Cross-backend candidate selection

Selecting between a `glproc` candidate and a `glcuda` candidate for the
same computation. This problem additionally requires whatever unit
reconciliation the previous section flags as open — subsystem (1)/(2)'s
costs and subsystem (3)'s costs are not stated in this directory's inputs
as commensurable quantities. A cost model addressing Problem A (same-
backend, same physical substrate) is not thereby shown to address Problem
B (cross-backend, different physical substrates); ARTX04/ARTX05 must treat
these as separate claims requiring separate evidence.

Both problems share the same *decision-agreement* success criterion
stated above (does `argmin` over estimated `m(P)` match `argmin` over
measured `m(P)`), but the estimator each requires operates over different
physical facts, and a cost model correct for Problem A carries no
established guarantee for Problem B.

---

## Exit Criteria (restated per this document's own requirement)

This document is complete because a reader can now state, without
ambiguity: the decision point is Step 5 of GATE's algorithm (`argmin
𝒞(P,w)` over an already-validated `𝒫valid`); the constraint on solving it
is obtaining `m(P)` without executing `P`; the success criterion is
decision-agreement with measured `m(P)` under the policy's actual weight
vector; the decision itself is not resource-bound; and any solution must
be evaluated separately for Problem A (same-backend) and Problem B
(cross-backend) rather than as one undifferentiated question.

---

## References

- ARTX01-Verified.md (physical vs. software-abstraction basis for the
  "not resource-bound" claim, and the three-subsystem identification)
- ARTX02-Observation.md (OBS-01 through OBS-04, OBS-07 through OBS-09 —
  the measured facts Problem A's worked instance and the subsystem
  descriptions draw on)
- `architecture/GATE/GATE-algorithm.md` (the seven-step sequence this
  problem is Step 5 of, and the empirical 1.4–8.2 µs decision-cost finding
  supporting the "not resource-bound" claim)
- ARTX04-Theory.md (what already exists against this problem statement)
- ARTX05-Gap.md (where existing solutions fall short of it)
