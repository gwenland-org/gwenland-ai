# Optimization Playbook

Reusable rules derived from the investigations in this directory. Each rule has a rationale and a pointer to the evidence behind it.

---

## Rules

### Rule 1 — Never trust isolated benchmarks alone

An isolated benchmark proves a kernel is faster in isolation. It does not prove production will improve. Always follow an isolated win with a production A/B before making any decision.

Evidence: Row Tile 2× GMAC/s isolated → +0.8–1.1% production (noise). VNNI-512 +20–26% GMAC/s isolated → +0.3–1.1% production (noise). Source: `observations.md`.

---

### Rule 2 — Always benchmark production before reporting a win

The only number that counts is the production end-to-end throughput (tok/s) measured with glbench on the full model, not an isolated kernel probe. All intermediate metrics are hypotheses until the production A/B confirms them.

---

### Rule 3 — Repeat ambiguous measurements before concluding

A single A/B is not sufficient if the result is within noise or changes sign between runs. Repeat in a fresh session. If the direction is inconsistent across two independent runs, the effect is below the detection threshold of this measurement setup.

Evidence: VNNI-512 run 1 showed +0.3%, run 2 showed +1.1% — both within noise. The inconsistency across runs was itself a signal to discount the result.

---

### Rule 4 — Compute the share bound before writing optimization code

`maximum_gain = stage_share × (1 − 1/expected_speedup)`. If this is below ~0.5% (the noise floor on this machine at 5 iterations), the optimization cannot produce a measurable production gain regardless of how good the kernel is.

Evidence: Bias/Residual SIMD. 0.085% share × 0.64 = 0.054% maximum gain. Not pursued.

---

### Rule 5 — Prefer removing work over accelerating work

Work elimination (Category 1 in the taxonomy) produces reliable production gains in all bottleneck regimes. Work acceleration (Category 2) is bottleneck-regime-dependent and frequently neutral in practice.

Evidence: RoPE cache (work elimination): +1.4–5.1% confirmed production gain. VNNI-512, Row Tile (work acceleration): neutral in production.

---

### Rule 6 — Do not optimize outside the measured bottleneck

Check the bottleneck regime of the target stage before designing the optimization. If the stage is bandwidth-bound, a compute speedup will not help. If the stage is compute-bound, a bandwidth reduction will not help.

Evidence: lm_head is bandwidth-bound. All prefill FFN stages are not-bandwidth-bound (compute-bound). Applying the same kernel technique to both regimes produces mixed results.

Source: `bottleneck_model.md`.

---

### Rule 7 — Use stage-level telemetry, not just total throughput

A global neutralization of a real stage-level gain is detectable only by looking at individual stage times. Always check `telemetry.decode.stages` before concluding that an optimization has no effect — it may have a real, stage-specific effect that is being cancelled by a regression elsewhere.

Evidence: Row Tile produced ffn_down −8.2% and lm_head +10.3% in the same run. Without stage telemetry, both would be invisible behind the +1.1% net decode number.

---

### Rule 8 — Test PPL before shipping any kernel or format change

Kernel changes can alter numerical behavior, especially when accumulator widths, reduction orders, or quantization paths change. A kernel that improves throughput but silently degrades PPL is a production regression.

Evidence: Q8_0 repack adds ~9% to the PPL gap vs llama.cpp. The decision to use Q8_0 was correct (GATE validates that it wins on this hardware), but the PPL cost is real and must be tracked.

---

### Rule 9 — Instrument unknown stages before optimizing around them

Stages without bandwidth measurements (`fixup`, `ffn_downq`, `serial`) account for 13–14% of prefill time and 8% of decode time. Their bottleneck regime is unknown. Any optimization analysis that does not account for them has a blind spot.

Source: `bottleneck_model.md` (Unknown Stages section).

---

### Rule 10 — A/B pairs must share the same bandwidth-ceiling measurement

On this machine, session-to-session bandwidth varies up to 19%. An A/B that compares runs from different sessions may see ceiling-efficiency differences that are pure measurement noise. Always run both baseline and candidate in the same glbench session.

Evidence: VNNI-512 candidate session returned 23.3 GB/s vs baseline's 31.1 GB/s, producing a spurious efficiency jump. Source: `anti_patterns.md` AP-05.

---

### Rule 11 — Per-stage wins require per-stage dispatch; do not use global flags

A global flag cannot capture a win that is real only for a subset of stages. The GATE architecture exists to dispatch per op-family. Use it.

Evidence: Row Tile implemented as global flag. ffn_down improved, lm_head regressed, net neutral. Source: `patterns.md` P-08.

---

### Rule 12 — Prefer GATE-based dispatch over env-var flags for all kernel variants

Env-var flags (`GLPROC_ROW_TILE=1`, `GLPROC_VNNI512=1`) are permanent technical debt: they cannot reason about per-stage bottlenecks, they are invisible to the calibration framework, and they cannot be composed. Route all kernel variants through GATE's Planner/CandidateSource.
