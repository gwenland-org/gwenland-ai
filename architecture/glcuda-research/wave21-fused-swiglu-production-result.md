# glcuda Wave 21 - fused SwiGLU production result

- notebook: `wave21-fused-swiglu-v1`
- notebook SHA-256:
  `6942d5105f51cf0d4e6b61cded439201a4d02a0e5db2f437e6665d0d32948cba`
- embedded Wave 21 patch SHA-256:
  `7f01ed0e154cd76312b27af2a436499ab5559fe07c1eaba713717126c8c84d5b`
- reconstructed base revision: `3bce8dd7b8aaa2765855ab927c611b54981f9241`
- GPU: Tesla T4, compute capability 7.5, driver 580.159.04
- Kaggle decision version: 2
- host gate before upload: 64/64 `glcuda` library tests

## Question and registered gate

Wave 20 moved the pinned 244-token production workload from 9,010.2 to
10,242.4 tok/s by replacing qk4 attention with fused compensated-MMA
attention. Its remaining current-stack cost is dominated by dense prefill
GEMMs and their FFN glue.

Wave 21 asks whether one CTA can compute matching gate and up rows, apply the
retained SwiGLU arithmetic, quantize the result to the exact Q8_0 image and
feed `ffn_down` without materialising two f32 intermediates or launching the
separate glue kernel. Both production arms retain all Wave 20 settings; the
only intended causal variable is `GLCUDA_FUSED_SWIGLU=1`.

The gate was fixed before the device run:

1. the candidate module must assemble for sm_75 with zero stack and spills;
2. the CUDA driver must report at least one active 512-thread block per SM;
3. candidate Q8 bytes and f32 scale bits must exactly match the retained path
   at full-64, ragged-17 and production 244-token shapes;
4. the complete CUDA parity suite and exact 50/50 token oracle must pass;
5. decode may not regress by more than 5% in any pair and median session-tail
   latency may not regress by more than 5%; and
6. retention requires at least +5% median prefill throughput with every one of
   four position-balanced pairs positive. A retention result would require a
   second byte-identical notebook run.

## Candidate design

The experimental `gl_gemm_mma_q8_fused_swiglu` kernel uses one 512-thread CTA
for 64 paired gate/up outputs across a 64-token slab. Sixteen warps preserve
the retained K32 dequantisation and integer-MMA order. Gate and up results are
exchanged through a 16,384-byte shared tile; gate warps apply the retained SiLU
operation order, and the CTA emits the same Q8_0 byte/scale layout consumed by
`ffn_down`.

Runtime policy remains explicit and reversible. The candidate requires sm_75,
the retained Wave 12 B-stage image, the 2-D GEMM grid, a stacked Q8 gate/up
matrix, dimensions divisible by the kernel tile, and the opt-in environment
flag. Every failed guard keeps the old two-GEMM-plus-glue path. A once-per-
process `[glcuda-ffn]` record makes a production launch observable rather than
inferring it from the configured flag.

## Validation reached

The final local reconstruction passed:

- `cargo check -p glcuda --example wave21_fused_swiglu --locked`;
- 64/64 `cargo test -p glcuda --lib --locked` tests;
- notebook Python syntax validation for all three code cells;
- byte-exact decompression of the embedded 48,604-byte Wave 21 patch; and
- `git apply --reverse --check` plus `git diff --check` on the reconstructed
  historical stack.

Kaggle version 1 stopped at a harness-only structural predicate. The predicate
searched for conversion opcodes across the whole PTX module and therefore
mistook an earlier indexing conversion for quantizer order. Version 2 scopes
the check to the SiLU and Q8 post-processing blocks. The kernel patch and its
SHA-256 did not change between versions.

On Kaggle version 2, the retained sm_75 module assembled cleanly:

| retained entry | registers | static shared | stack | spill stores | spill loads |
|---|---:|---:|---:|---:|---:|
| `gl_gemm_mma_q8_bstage` | 49 | 9,728 B | 0 B | 0 B | 0 B |
| `gl_attn_mma4_fused_f32` | 48 | 4,096 B | 0 B | 0 B | 0 B |

The separate Wave 21 module did not assemble. `ptxas` returned 255 with this
diagnostic:

```text
glcuda_wave21_sm75.ptx, line 83; error: Special register argument not allowed
for instruction 'shl'
ptxas fatal: Ptx assembly aborted due to errors
```

Line 83 attempts `shl.b32 %r11, %ctaid.x, 6`. On sm_75 PTX, the special CTA-ID
register cannot be supplied directly to that shift form; it must first be
moved into a general register. This is a bounded source defect, but changing it
after the compiler gate would create a new candidate and therefore belongs to
a fresh wave.

## Production result and decision

**COMPILE-GATE REJECT.** No Wave 21 production throughput number exists. The
candidate never produced a cubin, so driver occupancy, direct device parity,
the full CUDA parity suite and production A/B were intentionally not run.
Reporting a projected or host-only number here would violate the registered
measurement gate.

The retained production number therefore remains Wave 20's reproduced
**10,242.4 tok/s** at the pinned workload. This result neither validates nor
invalidates the fused-SwiGLU architecture; it rejects this exact PTX image
before performance measurement. The next admissible experiment is a fresh
wave with the special-register move corrected, the same resource/parity gates,
and no relaxation of the production threshold.
