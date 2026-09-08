# Wave 78 - fused compensated-MMA AV implementation gate

## Intended deliverable

Integrate the reproduced Wave 75 row-major compensated-MMA AV mapping into an
opt-in production-shaped attention entry without changing the retained
`gl_attn_mma4_regq_fused_f32` baseline. This wave is an implementation and host
contract gate. It does not claim device correctness or throughput.

## Implementation

The new sm_75 entry is `gl_attn_mma4_regq_avmma_fused_f32`. It clones the
retained Wave 48 Q staging, compensated-MMA QK, causal mask and softmax. Only
the final AV schedule changes:

1. Each of the four warps normalizes its four owned score rows in place.
2. One CTA barrier publishes all sixteen probability rows.
3. Every warp computes a 16-column output slice as two adjacent N8 fragments.
4. Four-product f16 compensation is applied to both P and V.
5. V is consumed directly from the retained `[kv_head, K, 64]` cache layout.

The path adds no allocation or permanent cache image. It reuses Wave 48's
dynamic score allocation, including the existing 4 KiB minimum required while
Q is staged. Ragged final query tiles and each row's distinct causal prefix are
predicated before shared probability loads and output stores.

The retained entry remains byte-for-byte present and is still selected unless
all three flags are enabled:

```text
GLCUDA_ATTN_MMA4=1
GLCUDA_ATTN_MMA4_REGQ=1
GLCUDA_ATTN_MMA4_AV=1
```

Runtime telemetry names the candidate `mma4-regq-avmma` and reports
`attn_mma4_av=true` in the process contract.

## Validation added

- The sm_75 structural test isolates the new entry and pins 40 static
  `m16n8k8` MMA sites: 32 retained QK sites plus eight AV sites.
- The candidate has four CTA barriers: the three retained register-Q barriers
  plus one probability-publication barrier.
- All 32 register-Q shared loads remain present.
- The GPU parity case now launches the new entry at:
  - Qwen production shape: 14 heads, 2 KV heads, base 0, 244 tokens;
  - ragged strided-Q shape: 4 heads, 2 KV heads, base 5, 17 tokens;
  - short tail shape: 2 heads, 1 KV head, base 7, 3 tokens.
- `wave78_mma_av_attention` provides an interleaved direct A/B against retained
  `mma4-regq`, reports compiler occupancy, max error, RMS and median latency,
  and enforces the registered `1e-5` numeric and 1.10x direct speed gates.

## Local results

Windows host, no CUDA device:

- baseline before edits: `cargo test -p glcuda --lib --locked` - 64/64 pass;
- after edits: the same suite - 64/64 pass;
- serial parity target: 33/33 pass in 0.02 s; this confirms compilation only,
  because no CUDA device executed the kernels;
- release A/B harness build: pass;
- PTX byte audit: ASCII only, LF only, zero CR bytes;
- `git diff --check`: pass.

The two existing `unused_mut` warnings in `glproc/src/loader.rs` remain
unchanged and out of scope.

## Decision and remaining risk

**HOST IMPLEMENTATION PASS; DEVICE EVIDENCE REQUIRED.** The candidate is wired
far enough for a controlled T4 decision while the retained production path and
fallback remain intact.

This result does not establish that `ptxas` accepts the entry, that it avoids
spills, that the new barrier and normalization pass net-win, or that its output
meets tolerance on hardware. The next wave must use the hard
`NvidiaTeslaT4` selector to:

1. assemble the full module and archive registers, stack, spills and barriers;
2. run the serial parity suite on the three registered shapes;
3. run the interleaved direct fused-attention A/B twice.

Only a passing device gate licenses a production `glbench` A/B. The 15,000
tok/s goal remains open.
