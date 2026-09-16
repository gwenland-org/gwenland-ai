# Wave 125: N16-prefetch fused SwiGLU design gate

## Frozen hypothesis

Wave 125 tests whether the retained Wave 88 N16 register-prefetch mainloop can
finish the paired gate/up projection directly as the Q8_0 activation consumed
by `ffn_down`. This is not a retry of Wave 21's 512-thread schedule. One
256-thread CTA keeps the retained eight-warps, M64, K32 order and two N8 MMA
fragments, but interprets the fragments as matching gate/up columns. After the
K loop, the CTA reuses its 16 KiB shared image for the exact retained SiLU and
Q8_0 epilogue.

The candidate is opt-in and may run only for the pinned production call:

- sm_75, Q8_0 B-stage weights and the Wave 88 N16-prefetch path;
- stacked gate/up shape `9728 x 896`, hidden dimension `4864`;
- exactly 244 prefill rows and the retained two-dimensional grid;
- hidden divisible by 128 and input dimension divisible by 32.

Every other call retains Wave 123's stacked N16-prefetch GEMM followed by its
no-store SwiGLU+Q8 kernel.

## Frozen implementation contract

- 256 threads and 76 x 4 CTAs for the pinned 244-token prompt;
- each warp owns eight final hidden columns across 64 token rows;
- N8 fragment zero reads gate rows and fragment one reads matching up rows;
- K32 traversal, f16 scale conversion and f32 accumulation order match Wave 88;
- the 16 KiB shared allocation aliases mainloop operands with the post-loop
  64-token x 64-hidden f32 epilogue slab;
- SiLU, max reduction, scale, reciprocal, rounding, clamp and Q8 stores match
  the retained PTX instruction order;
- no intermediate gate/up f32 global stores and no separate SwiGLU launch.

## Correctness and resource gates

1. Existing host tests pass before and after the change; structural tests prove
   the opt-in guard, retained fallback, entry parameters, MMA count and shared
   alias bound.
2. T4 `ptxas -v` reports zero stack, zero spill stores, zero spill loads,
   at most 80 registers and exactly 16,384 bytes static shared memory.
3. The driver occupancy query reports at least three resident CTAs per SM for
   the 256-thread launch.
4. Direct parity compares candidate Q8 bytes and f32 scale bits against Wave
   123 for full, ragged and production-sized inputs. Every compared byte and
   scale bit must match.
5. The CUDA parity suite passes and every production arm matches the exact
   token oracle.

Any compile, resource, launch, parity or oracle failure stops the wave.

## Production gate

The retained and candidate arms use one release binary, pinned T4, pinned GGUF,
244 prompt tokens, seed 42 and the complete retained Wave 123/Wave 118 feature
stack. Only the Wave 125 opt-in changes.

- two counterbalanced orders, at least 20 measured samples per arm per order;
- every order must improve median prefill throughput;
- retain only at >=4% aggregate prefill throughput with no decode regression
  worse than 5% and no tail-latency regression above 5%;
- 2-4% with every order positive is measurable but not retained;
- reject below 2% or any negative order;
- reaching the project goal additionally requires verified production median
  throughput of at least 15,000 prefill tok/s.

Kernel timing explains the outcome; only unprofiled production wall time makes
the retention decision.
