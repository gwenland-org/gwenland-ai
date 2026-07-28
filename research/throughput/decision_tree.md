# Decision Tree

A practical workflow for evaluating whether an optimization is worth pursuing.

---

## Entry Point

You have an optimization idea. It may come from a profiler, a code review, an architectural comparison (Percival), or a benchmark anomaly.

---

## Step 1 — Compute the share bound

Before writing any code, answer: **what share of production wall-clock time does this stage occupy?**

Use `benchmarks/full-bottleneck-e2e.json` (decode stage shares) or run `glbench run --kind decode` with `GLPROC_PROFILE=1` to get a fresh stage breakdown.

Then estimate: `maximum_gain = share × (1 − 1/expected_speedup_ratio)`

If `maximum_gain < noise_floor (~0.5–1%)`:
→ **REJECT**. The optimization cannot be measured, let alone shipped. Stop here.

If `maximum_gain ≥ noise_floor`:
→ Continue to Step 2.

**Example:**
- Bias/Residual SIMD: share 0.085%, speedup 2.81×. Maximum gain = 0.085% × 0.64 = 0.054%. Below noise floor. → REJECT.
- lm_head optimization: share 22.4%, speedup 10%. Maximum gain = 2.2%. Above noise floor. → Continue.

---

## Step 2 — Check bottleneck regime for this stage

Look at the stage's verdict in `full-bottleneck-e2e.json`:

- **Bandwidth-bound** (lm_head is the only confirmed example): memory traffic reduction can help. Compute acceleration cannot.
- **Not-bandwidth-bound** (all prefill stages): compute acceleration can help. Memory traffic reduction at this stage is not the bottleneck.
- **Indeterminate** (most decode stages): unknown. Need more information before committing.

If the optimization does not match the bottleneck regime:
→ **PAUSE**. Collect more data (Step 3). Do not assume the regime.

If the optimization matches:
→ Continue to Step 4.

---

## Step 3 — Roofline and memory traffic measurement

If the stage regime is indeterminate or unknown:

1. Run `glbench run --kind decode` with full telemetry.
2. Inspect `ceiling_frac` and `intensity_flop_per_byte` for the target stage.
3. If `ceiling_frac > 0.85` and `intensity_flop_per_byte < 5`: bandwidth-bound → memory traffic reduction.
4. If `ceiling_frac < 0.2` and `intensity_flop_per_byte > 50`: compute-bound → compute acceleration.
5. If neither: indeterminate.

For indeterminate stages, the only reliable next step is an A/B with the candidate change. Do not skip to Step 4 based on theory alone.

---

## Step 4 — Build and measure isolated

If the change targets a kernel (SIMD width, accumulator count, loop structure):
1. Add an isolated bench (see `glproc/benches/`).
2. Run the bench. Record GMAC/s baseline and candidate.
3. **Do not stop here.** Isolated results are necessary but not sufficient.

If the change eliminates work (table cache, memoization, deduplication):
- Skip the isolated bench step if the redundancy can be directly counted (e.g., "384× sin/cos per step" is countable). Proceed directly to Step 5.

---

## Step 5 — Run production A/B

```
glbench ab --engine glproc --model <model> \
  --kind decode --warmup 1 --iters 5 \
  --baseline <env=off> --candidate <env=on>
```

Rules:
- Both runs must be in the same session (same bandwidth-ceiling probe conditions).
- Use 5 measured iterations minimum.
- Compare mean decode tok/s and the CI95 range.

If `candidate_mean - baseline_mean < baseline_std_dev`:
→ **NEUTRAL**. The result is within noise. Do not report as an improvement.

If `candidate_mean - baseline_mean > 2 × baseline_std_dev` (or CI95 ranges do not overlap):
→ **LIKELY REAL**. Continue to Step 6.

---

## Step 6 — Repeat once

Run a second A/B in a fresh session (new terminal, re-probe bandwidth ceiling).

If the direction is consistent in both runs:
→ **CONFIRMED**. Proceed to Step 7.

If the direction flips or one run is neutral:
→ **AMBIGUOUS**. Do not ship. Consider whether the effect is stage-specific (Step 7b) or genuinely absent.

---

## Step 7a — Check stage-level breakdown

Even if the production total is neutral, check individual stage times:

1. Run `glbench run --kind decode` with telemetry before and after.
2. Check which stages improved and which regressed.
3. If a specific stage improved significantly (> noise, say 5%+) but another regressed:
   → The optimization is stage-specific. See Step 7b.

---

## Step 7b — Per-stage dispatch candidate

If the optimization helps stage A but harms stage B:

1. Document the stage-level breakdown in `observations.md`.
2. Mark the optimization as a "per-stage candidate" in `future_candidates.md`.
3. The correct fix is per-op-family GATE dispatch, not a global flag.
4. Do not ship as a global default.

---

## Step 8 — PPL check

Before any kernel or format change ships to production:

1. Run `glbench ppl --engine glproc --model <model>` baseline.
2. Apply the change and re-run.
3. If PPL increased by more than 2%: **RED FLAG**. Do not ship without understanding why.

---

## Summary Flow

```
Idea
 │
 ▼
Share bound (Step 1)
 │ < noise floor → REJECT
 │ ≥ noise floor
 ▼
Bottleneck regime (Step 2)
 │ matches? → Step 4
 │ unknown? → Roofline (Step 3) → Step 4
 ▼
Isolated bench (Step 4, if kernel change)
 ▼
Production A/B, 5 iters (Step 5)
 │ within noise → NEUTRAL
 │ above noise
 ▼
Repeat in fresh session (Step 6)
 │ consistent → CONFIRMED
 │ flip/neutral → AMBIGUOUS → stage breakdown
 ▼
Stage breakdown (Step 7a)
 │ uniform → ship
 │ stage-specific gain+regression → per-stage candidate (Step 7b)
 ▼
PPL check (Step 8)
 │ PPL safe → SHIP
 │ PPL degraded → INVESTIGATE before shipping
```
