# Anti-Patterns

Optimization traps observed or inferred from this investigation. Each includes what went wrong, and what to do instead.

---

## AP-01 — Chasing isolated GMAC/s as the primary optimization metric

**What happens:** You build a kernel that runs faster in isolation. You benchmark it with a probe. You report a 2× GMAC/s improvement. You ship it. Production throughput does not move.

**Why:** Isolated benchmarks measure a single kernel against a single bottleneck (compute or bandwidth). Production workloads are a pipeline of stages. The bottleneck rotates. A kernel that beats the compute ceiling in isolation may run into the bandwidth ceiling the moment it sits next to lm_head.

**Observed:** Row Tile produced 2× isolated GMAC/s and +0.8–1.1% production improvement (noise). VNNI-512 produced +20–26% isolated GMAC/s and +0.3–1.1% production improvement (noise).

**Instead:** Always run a production A/B immediately after an isolated win. If the production delta is within noise, the isolated result is not actionable. See the decision tree.

Source: `benchmarks/row-tile-ab-*.json`, `benchmarks/vnni512-ab-*.json`.

---

## AP-02 — Assuming wider SIMD always wins

**What happens:** You replace a 256-bit inner loop with a 512-bit inner loop. You expect 2× throughput. Production is flat.

**Why:** On the i3-1115G4 (Tiger Lake), 512-bit AVX-512 instructions split into two 256-bit micro-ops (Intel "splitting"). The throughput per cycle is the same as AVX2 at full-width. Additionally, if the bottleneck is memory bandwidth rather than compute, adding more compute throughput per cycle does not help.

**Observed:** VNNI-512 ran on the `avx2` simd_path in all production runs. The kernel was compiled but the dispatch path did not select it — or the selection was identical in performance. No production improvement was recorded in either repeat.

**Instead:** Check that the kernel is actually being dispatched before benchmarking. Confirm via `backend.simd_path` in glbench telemetry. Also check the roofline: if the stage is bandwidth-bound, wider SIMD is irrelevant.

Source: `benchmarks/vnni512-ab-baseline.json`, `benchmarks/vnni512-ab-candidate.json` (`backend.simd_path: "avx2"` in both).

---

## AP-03 — Using a global flag for a stage-specific win

**What happens:** You discover that an optimization helps ffn_down. You implement it as a global flag that applies to all stages. You benchmark. Production is neutral or negative because another stage (e.g., lm_head) regresses under the same flag.

**Observed:** Row Tile implemented as `GLPROC_ROW_TILE=1`. Applied globally. ffn_down improved 8.2%. lm_head regressed 10.3%. Net: neutral.

**Instead:** Stage-specific wins require stage-specific dispatch. The GATE `Planner`/`CandidateSource` architecture is built for exactly this: select the kernel per op-family, not per process.

Source: `benchmarks/row-tile-ab-baseline.json`, `benchmarks/row-tile-ab-candidate.json`.

---

## AP-04 — Optimizing outside the measured bottleneck

**What happens:** You optimize a stage that contributes a small share of total time. The absolute wall-clock savings are too small to measure, let alone exceed noise.

**Observed:** Bias/Residual SIMD produced a real 2.81× isolated speedup. The operation occupies 0.085% of decode wall-clock. Even with perfect optimization, the maximum possible production gain is 0.054% — below the measurement noise floor on this machine.

**Instead:** Use P-04. Compute the share before writing a single line of optimization code. If `share × (1 − 1/speedup_ratio) < noise_floor`, the work has no measurable production impact regardless of how good the kernel is.

Source: `notes/issues/glproc-throughput-gap-vs-llamacpp.md`.

---

## AP-05 — Trusting a single-session bandwidth measurement across sessions

**What happens:** You run a baseline in session A. You run a candidate in session B. The bandwidth ceiling probe returns different values in each session. You conclude the candidate has different ceiling efficiency. The difference is measurement noise.

**Observed:** VNNI-512 candidate run returned 23.3 GB/s bandwidth ceiling vs baseline's 31.1 GB/s. Candidate appeared to have 75.7% ceiling efficiency vs 56.4%. The gap was an artifact — the bandwidth probe on this machine shows 19% session-to-session variance.

**Instead:** Run A/B pairs in the same glbench invocation (`glbench ab`). This re-probes the ceiling at the start of each run within the same session, minimizing the thermal/scheduling gap between measurements.

Source: `benchmarks/vnni512-ab-baseline.json`, `benchmarks/vnni512-ab-candidate.json`, `benchmarks/post-gate-benchmark-report.md`.

---

## AP-06 — Inferring bottleneck from ceiling efficiency alone

**What happens:** You see a kernel is at 70% ceiling efficiency. You conclude it is bandwidth-bound. You design a memory-traffic reduction. You benchmark. Production is flat because the kernel was actually compute-stalled, not bandwidth-saturated.

**Observed:** The `attention` stage in decode shows 4.2 GB/s (6% of ceiling), which is very low bandwidth utilization — but glbench does not classify it as bandwidth-bound; it is "indeterminate." The hypothesis generated: "attention is stalled on something other than memory traffic." A memory-traffic reduction for attention would not help this stage.

**Instead:** Ceiling efficiency is one signal, not a complete bottleneck diagnosis. Combine it with arithmetic intensity. Check whether the stage's time share is large enough to matter (P-04). If unsure, measure memory traffic directly before designing a fix.

Source: `benchmarks/full-bottleneck-e2e.json`, `benchmarks/simd-check.json` (hypothesis: `"Decode bucket 'attention' holds 24% but reaches only 22% of ceiling — stalled on something other than memory traffic"`).

---

## AP-07 — Treating Q4_K→Q8_0 repack as a free precision swap

**What happens:** You load Q4_K_M weights, repack them to Q8_0 at load time for kernel-speed reasons, and assume the precision loss is negligible.

**Observed:** Q8_0 repack costs approximately 9% of the PPL gap vs llama.cpp (32.91 → 36.12 when switching repack on). That 9% is real, not negligible, though it is not the dominant factor in the total gap.

**Instead:** Count the PPL cost of format conversion explicitly. Do not assume repack is precision-neutral until it is measured.

Source: `notes/issues/glproc-precision-gap-vs-llamacpp.md`.
