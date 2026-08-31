# glcuda Wave 24 - independent-MMA B-stage GEMM production result

- notebook: `wave24-mma2-gemm-v1`
- final notebook SHA-256:
  `7ebc09f1621f11c4543493ab857e47bab0122839b44df212f2b6b6a3b3c967de`
- embedded Wave 24 patch SHA-256:
  `b161faf1e0bc88688c320e423ef1b3f4ade1ebdf1850895910cb195e6d0e8ee8`
- reconstructed base revision: `3bce8dd7b8aaa2765855ab927c611b54981f9241`
- GPU: Tesla T4, compute capability 7.5, driver 580.159.04
- Kaggle decision version: 1
- model SHA-256:
  `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`

## Question and causal scope

Wave 24 asks whether the two adjacent `m8n8k16` instructions in every retained
K32 B-stage step lose time to their shared s32 accumulator dependency. The
candidate starts each K16 dot product from zero in a disjoint s32 pair, adds
the pairs in s32, then runs the retained conversion, scale multiplication and
f32 FMA sequence.

The candidate preserves the Wave 12 B-stage weight image, N64/N128 launch
selection, grid, 9,728-byte shared-memory image, barriers, operand loads,
integer dot product, per-K32 dequantisation order, f32 accumulation order and
stores. It is opt-in through `GLCUDA_GEMM_MMA2=1`.

Both production arms retain Wave 20 compensated-MMA attention. The rejected
Wave 22 fused-SwiGLU and Wave 23 K128 superstage remain disabled.

## Validation

The reconstructed historical stack passed locally:

- `cargo check -p glcuda --example wave24_mma2_gemm --locked`;
- 66/66 `cargo test -p glcuda --lib --locked` tests;
- structural assertions for 16 MMA instructions, eight s32 pair reductions,
  the retained two-barrier K32 loop and unchanged 9,728-byte shared image;
- Python syntax and `nbformat` validation for every notebook cell;
- byte-exact decompression of the embedded 39,703-byte Wave 24 patch; and
- `git apply --reverse --check`, cached apply check and `git diff --check`.

Kaggle version 1 passed every registered correctness and resource gate:

- 66/66 `glcuda` library tests;
- 33/33 full CUDA parity tests with the candidate selected;
- bit-exact f32 output at both production GEMM shapes and the K160 diagnostic;
- exact 50/50 token oracle in all eight production sessions;
- retained `mma4-fused` attention in both production arms; and
- `bstage-mma2` dispatch only in the candidate arm.

The generated archive's report heading says “Wave 23” because one cosmetic
title fragment survived the Wave 24 harness derivation. Its build identifier,
embedded patch, resource records, dispatch contract, file names and all gate
logic identify Wave 24 correctly. The executed notebook is retained unchanged
so its SHA remains the exact Kaggle evidence image.

The candidate keeps the retained residency tier:

| entry | registers | static shared | active blocks/SM at 256t | active blocks/SM at 512t | stack | spills |
|---|---:|---:|---:|---:|---:|---:|
| retained K32 B-stage | 49 | 9,728 B | retained tier | retained tier | 0 B | 0 B |
| independent MMA chains | 50 | 9,728 B | 4 | 2 | 0 B | 0 B |

The direct bit-exact screen was effectively flat:

| shape | retained | MMA2 | retained/MMA2 | MMA2 throughput delta | bit-exact |
|---|---:|---:|---:|---:|---|
| `ffn_gate_up` 9728x896x244 | 339.176 us | 341.628 us | 0.9928x | -0.72% | yes |
| `ffn_down` 896x4864x244 | 191.562 us | 190.114 us | 1.0076x | +0.76% | yes |
| ragged K160 128x160x17 | 5.170 us | 5.132 us | 1.0075x | +0.75% | yes |

Both production shapes remained above the frozen 0.95x hard-stop threshold,
so the production gate ran as registered.

## Production metrics

The fixed 244-token workload used four position-balanced pairs in orders A/B,
B/A, B/A and A/B, with 5 cold, 5 warmup and 10 measured iterations per arm.

| arm | median tok/s | median latency | median MAD | median max | decode tok/s |
|---|---:|---:|---:|---:|---:|
| retained K32 | 10,542.5 | 23.145 ms | 0.113 ms | 23.377 ms | 206.7 |
| independent MMA chains | 10,473.7 | 23.297 ms | 0.191 ms | 23.716 ms | 205.7 |

- median throughput delta: **-0.65%**;
- paired throughput deltas: **-2.292%, +0.854%, -0.013%, -1.562%**;
- worst paired delta: **-2.29%**;
- median session-max latency delta: **+1.45%**;
- every paired decode delta remained above the -5% floor; and
- all token-oracle comparisons were exact.

## Decision and interpretation

**REJECT: -0.65%, with only one of four production pairs positive.** The
retained K32 B-stage kernel remains the production default. No product kernel
or dispatch change is promoted.

The experiment rules out a useful Tensor Core dependency-stall lever at this
tile shape. One additional register preserves full driver-reported residency,
yet the isolated production shapes stay within about one percent and the
end-to-end result is slightly negative. The two serial K16 MMA operations are
therefore not the missing arithmetic-engine gain identified after Wave 23.
Future GEMM work should not revisit accumulator-chain splitting on this T4
B-stage schedule.

Because the decision is REJECT, the protocol does not require a second
byte-identical confirmation run.
