# Future Candidates

Ranked optimization ideas for glproc CPU throughput. Ranked by expected impact × evidence quality.

Classification per entry:
- **Expected impact**: Low / Medium / High (relative to current tok/s gap vs llama.cpp).
- **Difficulty**: Low / Medium / High (engineering effort to prototype and validate).
- **Risk**: Low / Medium / High (risk to correctness or PPL).
- **Evidence**: what we already know that informs this candidate.
- **Required tooling**: what glbench or glproc capabilities are needed to validate it.
- **Status**: Hypothesis / Future work.

---

## FC-01 — Per-stage GATE dispatch for Row Tile (ffn_down only)

**Expected impact:** Medium. ffn_down is 20% of decode time. Row Tile produced a confirmed ffn_down −8.2% improvement in isolation. If applied only to ffn_down (avoiding lm_head), the expected net gain is ~1.6% decode.

**Difficulty:** Medium. Requires integrating Row Tile as a GATE `CandidateSource` with op-family scoping. The kernel exists behind `GLPROC_ROW_TILE=1`; the wiring into GATE's planner is the work.

**Risk:** Low. The kernel's correctness is already validated (runs in production behind flag). The risk is that the stage-level breakdown in the A/B varies — need a second run to confirm the ffn_down gain is reproducible in a fresh session.

**Evidence:** Row Tile produced ffn_down −8.2% in run 1, with consistent GMAC/s gains in both run 1 and run 2 at the stage level. lm_head regressed because it received the same kernel globally. Source: `benchmarks/row-tile-ab-baseline.json`, `benchmarks/row-tile-ab-candidate.json`.

**Required tooling:** GATE Planner extended with op-family kernel selection; `glbench ab` with stage-level telemetry comparison.

**Status:** Future work. Reference: `notes/issues/glproc-throughput-gap-vs-llamacpp.md`.

---

## FC-02 — RMSNorm / RoPE / softmax formula alignment with llama.cpp

**Expected impact:** High (for PPL). The current ~33 PPL point gap vs llama.cpp is unexplained after ruling out tokenization, BOS, and SIMD precision. It is narrowed to a formula mismatch in RMSNorm, RoPE, or softmax. Fixing this will not directly improve tok/s, but it will remove a correctness debt that currently disqualifies glproc for precision-sensitive use cases.

**Difficulty:** Medium. Requires a line-by-line comparison of glproc's `rms_norm_into`, `rope_apply`, and `softmax` against llama.cpp's `ggml_compute_forward_rms_norm_f32`, `ggml_compute_forward_rope_flt`, and `ggml_compute_forward_soft_max_f32` at commit `910196f`.

**Risk:** Low for PPL. Medium for throughput — formula changes may affect the fast path.

**Evidence:** PPL gap narrows from 36.12 → 32.91 when switching Q8_0 repack off, but 32.91 vs 24.78 (llama.cpp) remains unexplained. Forced scalar was not better (36.33), ruling out SIMD precision. BOS and tokenization ruled out. Source: `notes/issues/glproc-precision-gap-vs-llamacpp.md`.

**Required tooling:** Numeric unit test comparing glproc formulas to llama.cpp reference on a small synthetic input. Existing `glbench ppl` for before/after validation.

**Status:** Open investigation. Specifically not started as of 2026-07-26.

---

## FC-03 — Per-layer dequant cache (dequant once per step, reuse on recompute)

**Expected impact:** High (for training / layered loader). Not applicable to the current glproc inference path, but relevant to the training loop in `GWEN-219`/`GWEN-220`. Each training step re-dequantizes all 28 Q8_0 layers twice (forward + recompute). Caching each layer's F32 weights for the duration of one step removes half the dequant work. Estimated ~2× step speedup for the training path.

**Difficulty:** Low to Medium. Pure memoization; bit-identical weights. One layer of F32 at a time stays under the 8 GB RAM budget.

**Risk:** Zero (bit-identical). The only risk is RAM usage, which is bounded.

**Evidence:** Step wall-time is dequant-bounded on the training path (200–320s/step, measured in GWEN-219 dry-run). Dequant is confirmed as the dominant cost. Source: `Experimental/NewExperiment.md`.

**Required tooling:** Training loop instrumented with per-phase timing. Already has GWEN-219 baseline.

**Status:** Proposed (see `Experimental/NewExperiment.md`). Not yet prototyped.

---

## FC-04 — lm_head memory traffic reduction

**Expected impact:** Medium. lm_head is 22.4% of decode time and is the only stage classified bandwidth-bound (73.5% of ceiling). Any reduction in bytes read per call has a direct scaling factor on lm_head time.

**Techniques:** speculative decoding (skip lm_head on high-confidence tokens); tiled lm_head with KV-scale reuse; dynamic sparsity (only compute top-K logits).

**Difficulty:** High (speculative decoding is a pipeline change); Low–Medium (top-K logit computation).

**Risk:** Medium. Speculative decoding requires a draft model. Top-K logit computation changes the sampling path and must preserve sampling correctness.

**Evidence:** lm_head verdict: bandwidth-bound, 73.5% ceiling, 22.4% of decode. Source: `benchmarks/full-bottleneck-e2e.json`.

**Required tooling:** Stage-level A/B with glbench. PPL check after any logit-computation change.

**Status:** Hypothesis.

---

## FC-05 — ffn_gate_up memory traffic reduction (fused quantized GEMV)

**Expected impact:** Medium to High. ffn_gate_up is 35.5% of decode time (the largest single stage). Its bandwidth utilization is 68.2% of the ceiling — near bandwidth-bound territory. Reducing bytes read (tighter quantization, better cache reuse) or fusing the gate+up+SwiGLU into a single memory pass could improve this stage.

A single-pass fused SwiGLU kernel already exists (confirmed active: "Q8_0 fused-swiglu integer-dot"). The question is whether the memory traffic can be further reduced.

**Difficulty:** Medium. Requires profiling the actual memory access pattern of the fused SwiGLU kernel to find remaining redundancy.

**Risk:** Low (if staying with Q8_0); Medium (if changing format).

**Evidence:** ffn_gate_up share 35.5%, GB/s 22.4 (68.2% ceiling). Source: `benchmarks/full-bottleneck-e2e.json`. Fused SwiGLU already active; further reduction requires measuring current access pattern.

**Required tooling:** Memory traffic instrumentation for ffn_gate_up (currently `bytes_read` is measured; need cache hit/miss or access-pattern analysis, which requires OS counters not available without elevated privileges on Windows).

**Status:** Hypothesis. Blocked on memory access profiling tooling.

---

## FC-06 — VNNI-512 dispatch verification and retry on correct path

**Expected impact:** Low–Medium. VNNI-512 showed neutral production results, but the kernel may not have been actually dispatched — both baseline and candidate reported `simd_path: "avx2"`. If the 512-bit path was never actually selected, the A/B measured identical paths.

**Difficulty:** Low. Add a log line confirming which SIMD path the qdot kernel selected at runtime. Re-run the A/B with verified dispatch.

**Risk:** Low. The kernel is already validated for correctness.

**Evidence:** Both VNNI-512 A/B runs show `backend.simd_path: "avx2"`. Source: `benchmarks/vnni512-ab-baseline.json`, `benchmarks/vnni512-ab-candidate.json`.

**Required tooling:** A single log line in the qdot dispatch path. Then standard `glbench ab`.

**Status:** Hypothesis (the kernel may have been silently falling back to AVX2 in both arms of the A/B).

---

## FC-07 — Attention stage stall investigation

**Expected impact:** Unknown. The `attention` stage uses only 4.2 GB/s (13.4% of ceiling) for 5.9% of decode time. This is anomalously low bandwidth utilization. glbench's hypothesis: "stalled on something other than memory traffic." If the stall is a synchronization barrier or a serial dependency, removing it could recover 5–6% of decode time.

**Difficulty:** High. Requires identifying the stall cause, which requires instrumentation not currently available (per-call timing, cache miss rates). The attention path is in `glproc/src/attention.rs`.

**Risk:** Medium. Attention is correctness-sensitive (softmax, causal masking).

**Evidence:** `attention` stage: 5.9% share, 4.2 GB/s (indeterminate). `simd-check.json` hypothesis: stalled on non-memory cause. Source: `benchmarks/full-bottleneck-e2e.json`, `benchmarks/simd-check.json`.

**Required tooling:** Per-call timing within the attention stage; possibly perf counters (not available on this machine without admin/RAPL).

**Status:** Hypothesis. Low tooling readiness.
