# Patterns

Engineering patterns extracted from the investigations. Each must reference supporting evidence.

---

## P-01 — Eliminating redundant work consistently improves production

**Statement:** When an optimization eliminates computations that are genuinely redundant (i.e., the same result was computed multiple times and thrown away), the production improvement is real and crosses the noise floor.

**Evidence:**
- RoPE table cache eliminated 384× redundant `sin_cos()` calls per decode step. Decode improved +1.4% (run 1), +5.1% (run 2). The `fixup` stage shrank 87.9%. Both runs confirmed the direction.
- The Q4_K→Q8_0 repack at load time is another example: converting weights once and serving them repeatedly is faster than the reverse.
- Source: `benchmarks/rope-ab-baseline.json`, `benchmarks/rope-ab-candidate.json`, `benchmarks/rope-ab-baseline-r2.json`, `benchmarks/rope-ab-candidate-r2.json`.

**Contrast:** VNNI-512 and Row Tile do the same work faster (not less work). Neither produced consistent production gains. See P-02.

---

## P-02 — Doing the same work faster does not guarantee production gains when the bottleneck is mixed

**Statement:** Kernel-level speedups that reduce compute time for an already-fast kernel do not automatically transfer to production throughput when the pipeline has both compute-bound and bandwidth-bound stages.

**Evidence:**
- VNNI-512 improved ceiling efficiency in glbench's isolated measurement but produced +0.3% and +1.1% decode improvements in two 5-iteration production runs — both within noise.
- Row Tile produced 2× GMAC/s in isolation but +1.1% and +0.8% in production — both within noise.
- glproc's decode pipeline runs at 57% of the bandwidth ceiling. Multiple stages are classified "indeterminate" (neither clearly compute-bound nor bandwidth-bound). A kernel optimization that helps one stage is absorbed by another stage that does not benefit.
- Source: `benchmarks/full-bottleneck-e2e.json`, `benchmarks/vnni512-ab-*.json`, `benchmarks/row-tile-ab-*.json`.

---

## P-03 — Stage-level wins are masked by cross-stage cancellation

**Statement:** An optimization can produce a real, measurable win at the stage level and still produce zero or negative net production improvement when another stage regresses or dominates.

**Evidence:**
- Row Tile produced a real stage-level gain: ffn_down −8.2% in run 1. But lm_head regressed +10.3% in the same run. lm_head occupies 22–25% of decode time; ffn_down occupies 18–20%. The regression exceeded the gain in wall-clock terms and the net decode was +1.1% (noise).
- The stage shares confirm the arithmetic: a 10% regression in a 22.4% stage outweighs an 8% improvement in a 20% stage.
- Source: `benchmarks/row-tile-ab-baseline.json`, `benchmarks/row-tile-ab-candidate.json`.

---

## P-04 — Absolute contribution bounds the maximum achievable production gain

**Statement:** Before attempting to optimize any operation, compute its share of total wall-clock time. The theoretical maximum production gain is bounded by that share.

**Evidence:**
- Bias/Residual SIMD: 2.81× isolated speedup. Share of decode wall-clock: 0.085%. Maximum possible production gain: 0.085% × (1 − 1/2.81) = 0.054%. Below noise floor.
- ffn_gate_up is 35.5% of decode time. A 10% improvement there could yield up to 3.5% decode improvement — above noise floor.
- lm_head is 22.4% of decode time. A 10% regression there costs up to 2.2% decode — above noise floor and already observed in Row Tile.
- Source: `benchmarks/full-bottleneck-e2e.json`, `notes/issues/glproc-throughput-gap-vs-llamacpp.md`.

---

## P-05 — Prefill and decode have different bottlenecks and respond differently to kernel changes

**Statement:** An optimization that helps decode may not help prefill, and vice versa, because the two phases have different arithmetic intensities and different bottleneck regimes.

**Evidence:**
- Decode: all major stages classified as "indeterminate" or "bandwidth-bound". Arithmetic intensity: ~1.88 FLOP/byte for FFN/lm_head.
- Prefill: all major stages classified as "not-bandwidth-bound". Arithmetic intensity: 60–67 FLOP/byte (32–35× higher than decode). Prefill is clearly compute-bound.
- Row Tile: neutral decode, mixed-sign prefill across runs.
- VNNI-512: neutral decode, mixed-sign prefill across runs (run 2 prefill was heavily noisy due to a bandwidth-probe variance artifact).
- Source: `benchmarks/full-bottleneck-e2e.json`.

---

## P-06 — Isolated GMAC/s is not a proxy for production throughput

**Statement:** GMAC/s measured in an isolated kernel benchmark does not predict production throughput improvement. The two metrics can diverge by 2× or more with zero production gain.

**Evidence:**
- Row Tile: 2× isolated GMAC/s (strongest isolated result). Production decode: +0.8–1.1% (noise). Ratio: 100% isolated improvement → ~1% production improvement.
- VNNI-512: +20–26% isolated GMAC/s. Production decode: +0.3–1.1% (noise).
- RoPE cache: no GMAC/s improvement (work elimination, not acceleration). Production decode: +1.4–5.1% (above noise, confirmed across two runs).
- Source: all `benchmarks/*-ab-*.json`, `notes/issues/glproc-throughput-gap-vs-llamacpp.md`.

---

## P-07 — Session-to-session bandwidth variance invalidates single-session A/B comparisons

**Statement:** The measured bandwidth ceiling on this machine varies up to 19% between sessions due to thermal and scheduling noise. An A/B comparison that does not measure the ceiling in the same session can misclassify the bottleneck regime.

**Evidence:**
- VNNI-512 candidate was measured in a session where the bandwidth probe returned 23.3 GB/s vs the baseline's 31.1 GB/s. This made the candidate appear to have 75.7% ceiling efficiency vs baseline's 56.4% — but the higher efficiency was an artifact of a lower ceiling, not a kernel improvement.
- All A/B pairs in this investigation used separate glbench runs back-to-back in the same session. The bandwidth ceiling is re-measured at the start of each run. Values ranged 29.7–31.8 GB/s within sessions, 23.3–31.8 GB/s across the full investigation.
- Source: `benchmarks/vnni512-ab-baseline.json`, `benchmarks/vnni512-ab-candidate.json`.

---

## P-08 — Per-stage dispatch, not global flags, is required to capture stage-specific wins

**Statement:** When a kernel improvement benefits one stage but harms another (e.g., Row Tile helping ffn_down but hurting lm_head), a global process-level flag (`GLPROC_ROW_TILE=1`) cannot capture the win while avoiding the regression.

**Evidence:**
- Row Tile is currently implemented as a global env-var flag. When enabled, it applies the same kernel to all eligible stages, including lm_head (which regressed 10.3%).
- The GATE architecture (Planner + CandidateSource) is explicitly designed for per-op-family kernel dispatch. The row-tile `ffn_down`-only win is a concrete, evidence-backed candidate for GATE once revisited.
- Source: `notes/issues/glproc-throughput-gap-vs-llamacpp.md`, `architecture/percival/IMPLEMENTATION-PLAN.md` (IP-05).
