# glproc

**The CPU inference engine, and the numerical ground truth.**

## What level does it work at?

**Sub-tensor: SIMD lanes and quantisation blocks.** This is the lowest level in
the workspace. glproc does not think in tokens; it thinks in 32-lane AVX2
registers and 256-weight Q4_K superblocks, and a token falls out the far end.

| Level | Example |
|---|---|
| **Register** | `kernels/qdot/q8_0/avx2.rs` — `_mm256_dpbusd_epi32` accumulator chains |
| **Block** | `kernels/dequant/q4_k/` — 256-weight superblocks, 8 sub-block scales |
| **Row** | `par_matmul_qdot`, `row_dot_q8_packed8` — M=8 weight-reuse GEMM tiling |
| **Layer** | `attention.rs`, `moe.rs`, `runner.rs` — the forward pass |

**glproc is the oracle.** Every GPU backend is validated against it
tensor-by-tensor with explicit per-operation tolerances. A change that shifts
glproc's numerics does not just break glproc — it invalidates the parity
baseline for every other engine.

## Where it stands, measured

Reference machine: i3-1115G4 (2p/4l), Windows, AC power.

| Claim | Number |
|---|---|
| Decode vs llama.cpp | **21.5% behind** (2026-07-28, measured interleaved) |
| Prefill vs llama.cpp | **20.2% behind** (same session) |
| Perplexity, native Q4_K | **24.19 vs llama.cpp 24.78 ± 3.69 — parity** |
| Perplexity, default repack | 26.65 (documented Q4_K→Q8_0 trade) |
| Decode bandwidth | ~20.4 GB/s effective of a measured 29.4 GB/s ceiling (~69%) |

⛔ **This machine drifts ~24% between sessions.** Only compare runs from the
same session. Sequential A/B once claimed f64 was "17% faster", which is
impossible.

## Dependencies

**One.** `num_cpus`. Sixteen crates in the full tree, almost all inherited from
`glcore`.

Everything else — SIMD dispatch, the persistent thread pool, every quant
kernel, the KV cache — is hand-written against `core::arch`.

## `unsafe`

Present and justified: SIMD intrinsics behind runtime CPU-feature dispatch.
Every block carries a `// SAFETY:` comment naming the invariant and where it is
established. Intrinsic kernels are reachable **only** through the
`SimdStrategy` match — never call an AVX2 kernel directly.

## Optimisations that were tried and rejected — with numbers

Read `gl-agent-skills/cpu-skills/rejected-optimizations.md` before proposing a
speedup. Three separate ideas measured **2× in an isolated probe and neutral in
production**:

- **Native Q4_K kernels** — *lost 33%*. Nibble unpack is compute-bound, and the
  gap is identical in L2, so it is not memory. Repacking to Q8_0 is the correct
  trade on AVX2/VNNI-256.
- **VNNI-512** — +20–26% isolated, flat in production, two repeats.
- **Row-tile GEMM (16 accumulator chains)** — 2× GMAC/s isolated; `ffn_down`
  +8.9% but `lm_head` −9.4%, cancelling out.

What *did* work: the **RoPE table cache**. `rope()` was recomputing sin/cos 384×
per forward pass on Qwen2.5-0.5B. Bit-exact fix, +1.2–5.2% decode and +3.2–6.8%
prefill across two production A/B repeats.

## Build

```bash
cargo test -p glproc
cargo bench -p glproc            # probes; see bench-skills/measurement-discipline.md
GLPROC_THREADS=4 cargo run ...   # thread count override
```
