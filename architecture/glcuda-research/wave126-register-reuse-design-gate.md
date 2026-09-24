# Wave 126: fused SwiGLU register-reuse repair gate

## Frozen hypothesis

Wave 125 failed its first T4 gate because the fused N16-prefetch entry compiled
with one 32-bit spill: an 8-byte stack frame plus 4-byte spill store and load.
Wave 126 tests one repair only. The prefetch temporary `%rP_next` is dead after
the mainloop, so the fused setup and epilogue reuse it instead of declaring
`%r_fgu_tmp`.

No arithmetic, instruction order, predicate, buffer address, shared-memory
layout, launch geometry, dispatch guard, or production workload may change.
The retained fallback remains Wave 123's stacked N16-prefetch GEMM followed by
the no-store SwiGLU plus Q8 path.

## Frozen implementation contract

- retain the Wave 125 256-thread, M64, K32 and paired gate/up N8 schedule;
- retain one 16,384-byte aliased shared-memory image;
- remove `%r_fgu_tmp` and reuse `%rP_next` before its prefetch lifetime and
  after the final mainloop use;
- keep the exact production guard `(hidden=4864, in_dim=896, ntok=244)`;
- keep `GLCUDA_N16_FUSED_SWIGLU` opt-in and every unsupported shape on the
  retained path.

## Ordered gates

1. Existing host tests and structural checks pass. The PTX remains ASCII/LF,
   contains no `%r_fgu_tmp`, and proves `%rP_next` is used in both the prefetch
   decision and fused quantization epilogue.
2. Tesla T4 `ptxas -v` reports zero stack, zero spill stores, zero spill loads,
   at most 80 registers, one barrier, and exactly 16,384 bytes static shared
   memory. Any stack or spill stops the wave.
3. Driver load succeeds and occupancy is at least three 256-thread CTAs/SM.
4. The serial CUDA parity suite passes. Direct candidate output must match the
   retained Q8 bytes and f32 scale bits exactly for the full production slabs
   and 52-row ragged tail before any timing is accepted.
5. Production wall-time uses one release binary and loaded model, seed 42,
   244 prompt tokens, five warmups per arm, two counterbalanced invocation
   orders, and 20 measured samples per arm per order.

## Retention authority

- both invocation orders must improve their production median;
- retain only at least 4% aggregate prefill throughput improvement, with no
  decode regression above 5% and no tail-latency regression above 5%;
- 2-4% with both orders positive is measurable but not retained;
- below 2% or any negative invocation order is rejected;
- the project goal is achieved only by a verified production median of at
  least 15,000 prefill tok/s.

Kernel timing and telemetry explain a result. Only unprofiled production wall
time decides retention.
