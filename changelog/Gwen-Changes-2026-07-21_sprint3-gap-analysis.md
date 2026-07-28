**Type:** Gap analysis — Veritas Prima Sprint 3 (Prefill GEMM Tiling) halted before Wave 1.
**Status:** No code changed. Reporting back to JinXSuper for a decision on the real next lever.

---

## Executive Summary

Sprint 3's brief asks for a tiled prefill GEMM kernel, citing
`rejected-optimizations.md`'s note that "tiling remains legitimate for
prefill GEMMs" as the one untried lever against the 141.2 vs 211.99 tok/s
gap (glproc vs llama.cpp, Qwen2.5-0.5B Q4_K_M, compute-bound at 68% ceiling
efficiency).

Before writing Wave 2's kernel, I read the current prefill path
(`glproc/src/runner.rs::step_chunk`, `glproc/src/threading.rs`,
`glproc/src/kernels/qdot/q8_0/avx2.rs`) end to end. **The tiling the brief
asks for is already shipped and already the production path** — it is not
untried. Stopped here rather than re-implementing it under a new name.

## What's actually in the code

`par_matmul_qdot` (`threading.rs:693`), used by `step_chunk` for every Q8_0
weight projection in prefill (QKV, wo, gate/up, down, lm_head):

- Packs `PREFILL_CHUNK` (32) activations into panels of **M_tile = 8**
  tokens (`threading.rs:725-760`), block-interleaved into one sequential
  stream per weight row.
- Each weight row is loaded and decoded (f16 scale, sign-prep) **once**,
  then dotted against all 8 panel activations via 8 independent FMA
  accumulator chains — `row_dot_q8_packed8`
  (`kernels/qdot/q8_0/avx2.rs:109-145`).
- Ragged/small-batch and non-Q8_0-wide fallback uses the same mechanism at
  G=4/2/1 (`row_dot_q8_xn::<G>`, `threading.rs:800-854`).
- AVX2 V-accumulation, `loadu` unaligned loads, scalar twin
  (`kernels/qdot/q8_0/scalar.rs`) — all per `avx2-simd.md` rules already.

This **is** M×K×N prefill-GEMM tiling: M (token batch) reuses the streamed
weight row exactly the way Wave 2's spec describes, K is the full row
streamed through registers per call, N is the row-chunk split across the
thread pool. The 68%-ceiling / 141.2 tok/s number in the brief is a
measurement *of this already-tiled path*, not of a naive per-token matvec
loop.

The only remaining untiled f32 `matmul::avx2::run` call is in
`attention.rs` for scores×V accumulation — attention math, explicitly a
Sprint 3 non-goal, not a weight-projection GEMM.

## Why Wave 1–3 didn't run

Writing a second "tiled prefill kernel" behind `tiled-prefill` would tile
the same axis (M, via weight-row reuse) that `par_matmul_qdot` already
tiles. Per `rejected-optimizations.md`'s anti-pattern note — matching
mechanism, not matching name, is what disqualifies a re-proposal — this
would be re-running a completed, currently-in-production experiment at the
user's expense, not a new lever. Checked `changelog/` for a prior
"tiled-prefill" attempt: none exists: this is a fresh finding, not a
previously-rejected idea returning.

## Open question for JinXSuper

The 33% gap vs llama.cpp is real and still unexplained by "tiling was never
tried." Candidate next levers, **none attempted yet, all needing their own
roofline check + production A/B before implementation**:

1. Panel width G=8 is register-bound (8 accumulators + weight regs in 16
   ymm). Whether restructuring the outer loop to amortize weight decode
   (`w_abs`/sign-prep) across all 4 panels of a `PREFILL_CHUNK=32` batch
   (instead of redoing it per 8-panel) saves anything — likely small, since
   the load itself isn't saved, only the decode.
2. N-dimension (output row) chunking currently splits `out_dim` evenly
   across `pool.n_threads()` with no row-blocking for L2 residency across
   panels — whether output-row tiling (distinct from the already-done
   M-tiling) has headroom.
3. Whether the 33% gap is actually in prefill's matmul stage at all, or in
   `p_fixup`/`p_attn`/`p_serial` (bias/RoPE/head-norm/quantize serial work
   per `Prof` in `runner.rs`) — no current profile breakdown was available
   for this exact baseline to confirm the matmul stage is where the gap
   lives.

Recommend profiling (`GLPROC_PROFILE=1`) the production prefill path first
to confirm *where* the remaining 33% actually sits before scoping the next
kernel change.
