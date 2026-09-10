# glcuda Wave 22 - repaired fused SwiGLU production result

- notebook: `wave22-fused-swiglu-v1`
- notebook SHA-256:
  `182b2a5a3070de26b0f2ad582dec6c819324a26a279aa3708eb3640c8f79c0ae`
- embedded Wave 22 repair patch SHA-256:
  `07c60277043e66874ddfdc36a0fcfa4057bdb6f8c7a73372064bd359411d6239`
- reconstructed base revision: `3bce8dd7b8aaa2765855ab927c611b54981f9241`
- GPU: Tesla T4, compute capability 7.5, driver 580.159.04
- Kaggle decision version: 2
- model SHA-256:
  `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`

## Question and causal scope

Wave 21's fused-SwiGLU image failed the compiler gate because it supplied the
special `%ctaid.x` register directly to `shl`. Wave 22 asks the same registered
performance question after one bounded PTX repair: move `%ctaid.x` into the
existing `%r11` general register, then shift `%r11`.

No math, named-register declaration, launch geometry, shared-memory layout,
dispatch rule, fallback, workload or decision threshold changed. Both
production arms retain Wave 20 fused MMA attention. The sole A/B variable is
`GLCUDA_FUSED_SWIGLU`.

## Validation

The reconstructed historical stack passed locally:

- `cargo check -p glcuda --example wave21_fused_swiglu --locked`;
- 64/64 `cargo test -p glcuda --lib --locked` tests;
- a structural assertion forbidding `shl.b32 %r11, %ctaid.x, 6;`;
- Python syntax validation for every notebook code cell;
- byte-exact decompression of the embedded 4,223-byte repair patch; and
- `git apply --reverse --check` plus `git diff --check`.

Kaggle version 1 cleared compiler, resource, library, full parity and direct
bit-exact gates, then stopped on a harness-only assertion which expected the
direct kernel-handle example to emit a `Runner` dispatch record. That example
intentionally bypasses `Runner`, so version 2 removed only that impossible
direct-example log assertion. The production `glbench` dispatch audit remained
mandatory, and the embedded kernel patch stayed byte-identical.

Kaggle version 2 passed all correctness and resource gates:

- 64/64 `glcuda` library tests;
- 33/33 full CUDA parity tests;
- exact 50/50 token oracle in every production session;
- exact Q8 bytes and f32 scale bits at full-64, ragged-17 and production-244;
- observed retained `mma4-fused` attention in both arms; and
- observed `fused-swiglu-q8` at `[hidden, in_dim, ntok] = [4864, 896, 244]`
  only in the candidate arm.

The candidate resource image is viable:

| entry | registers | static shared | active blocks/SM | stack | spill stores | spill loads |
|---|---:|---:|---:|---:|---:|---:|
| `gl_gemm_mma_q8_fused_swiglu` | 57 | 16,384 B | 2 | 0 B | 0 B | 0 B |

The direct exact-parity screen produced:

| shape | retained | fused | retained/fused | exact Q8/scales |
|---|---:|---:|---:|---|
| full-64 | 14.770 us | 14.384 us | 1.0269x | yes |
| ragged-17 | 12.123 us | 9.305 us | 1.3028x | yes |
| production-244 | 445.552 us | 449.910 us | 0.9903x | yes |

The production-shaped direct launch is effectively flat and is not used as
the production decision. The registered end-to-end A/B is authoritative.

## Production metrics

The fixed 244-token workload used four position-balanced pairs in orders A/B,
B/A, B/A and A/B, with 5 cold, 5 warmup and 10 measured iterations per arm.

| arm | median tok/s | median latency | median MAD | median max | decode tok/s |
|---|---:|---:|---:|---:|---:|
| retained SwiGLU | 10,310.6 | 23.666 ms | 0.153 ms | 24.074 ms | 205.2 |
| fused SwiGLU | 10,389.0 | 23.486 ms | 0.215 ms | 23.952 ms | 205.0 |

- median throughput delta: **+0.76%**;
- paired throughput deltas: **+0.58%, -0.08%, +1.61%, +1.37%**;
- worst paired delta: **-0.08%**;
- median session-max latency delta: **-0.51%**;
- all paired decode deltas remained above the -5% floor; and
- all token-oracle comparisons were exact.

## Decision

**WEAK: +0.76%, positive but inside the +5% retention band.** One of four
pairs is also slightly negative, so the every-pair-positive gate is not met.
The retained two-GEMM-plus-glue SwiGLU path remains the production default.

Because the decision is not RETAIN, the registered protocol does not require a
second byte-identical confirmation run. The experiment establishes that the
repaired fusion is correct, spill-free and resident at two blocks per SM, but
its production benefit on this workload is too small to justify the added
kernel and dispatch surface.
