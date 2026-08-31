# glcuda Wave 24 - two-chain B-stage MMA design gate

## Frozen question

Does breaking the retained B-stage kernel's two serial `m8n8k16` operations
per K32 block into two independent integer accumulator chains materially
improve production prefill throughput on Tesla T4?

The retained kernel issues the K[0..16] MMA into one s32 accumulator pair and
then issues K[16..32] into that same pair. The second instruction therefore
has a read-after-write dependency on the first. Wave 24 gives each instruction
its own zeroed s32 pair, adds the pairs in s32 after both MMAs, and keeps the
existing conversion, scale multiply, and f32 FMA sequence unchanged.

## Frozen causal scope

The candidate keeps the retained Wave 12 B-stage weight image, N64/N128 launch
selection, grid, 9,728-byte shared-memory layout, barriers, global/shared
loads, `m8n8k16` operands, integer dot product, per-K32 dequantisation, f32
accumulation order, stores, and all fallbacks. It changes only the dependency
graph of each adjacent MMA pair and is opt-in through
`GLCUDA_GEMM_MMA2=1`.

Both production arms retain Wave 20 fused compensated-MMA attention and leave
the rejected Wave 22 fused-SwiGLU and Wave 23 K128 superstage disabled.

## Gates, in order

1. Reconstruct the SHA-verified Wave 3 through Wave 23 research stack and
   apply one incremental Wave 24 patch.
2. Pass the historical 65 `glcuda` library tests plus Wave 24 structural and
   dispatch tests. Structural checks must prove 16 static MMA instructions,
   eight integer pair reductions, the retained 9,728-byte shared layout, and
   no K128-superstage instructions in the candidate entry.
3. Assemble the full sm_75 module with `ptxas -v`. Retained and candidate
   entries must have zero stack, zero spill stores and zero spill loads. The
   candidate must use at most 56 registers and exactly 9,728 bytes of static
   shared memory.
4. Ask the CUDA driver for occupancy at both legal launches. The candidate
   must retain at least four active blocks per SM at 256 threads and two at
   512 threads, both with zero dynamic shared memory.
5. Require bit-exact f32 output versus retained B-stage at `ffn_gate_up`
   (9728x896), `ffn_down` (896x4864), and a ragged output/K diagnostic. Direct
   timing is a screen; either production shape below 0.95x is a hard stop.
6. Run the complete CUDA parity suite serially with the candidate selected.
7. Run four position-balanced production pairs in orders A/B, B/A, B/A, A/B,
   with 5 cold, 5 warmup and 10 measured iterations per arm. Every session
   must match the glproc token oracle exactly, 50/50, and dispatch audit must
   observe `bstage-mma2` only in the candidate arm.
8. Retain only if median production prefill improves by at least 5%, every
   pair is positive, every paired decode delta is at least -5%, and median
   session-max latency regresses by at most 5%.

A retention result requires a second run of byte-identical notebook content.
Any compiler, resource, parity, oracle or hard-regression failure terminates
the wave without fix-forward. A harness-only failure may be corrected only
while the embedded kernel patch remains byte-identical.
