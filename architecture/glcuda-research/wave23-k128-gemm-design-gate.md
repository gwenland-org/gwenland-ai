# glcuda Wave 23 - K128 B-stage GEMM design gate

## Frozen question

Does grouping four retained K32 B-stage iterations into one K128 shared-memory
superstage materially improve the production prefill path on Tesla T4?

Wave 16 measured the retained B-stage mainloop at roughly 26% exposed staging
and showed that `ffn_down` pays 5.4 times as many barrier pairs as
`ffn_gate_up`. Wave 17's one-block register prefetch collected only 3.3-3.8%
because it kept the one-stage shared-memory schedule. Wave 23 tests the other
ranked mechanism: spend the 38,912-byte MMQ-class shared-memory budget so four
K32 blocks are resident before computation begins.

## Frozen causal scope

The candidate keeps the retained Wave 12 B-stage weight image, N64/N128 launch
selection, output grid, `m8n8k16` operands, integer accumulation, per-K32 f32
dequantisation and f32 FMA order. It changes only mainloop staging:

- shared A/xscale/B/wscale storage grows from 9,728 to 38,912 bytes;
- up to four consecutive K32 blocks are staged before one barrier;
- the four blocks are consumed in the original ascending order; and
- one barrier follows the group before those shared slots are overwritten.

The candidate is opt-in through `GLCUDA_GEMM_K128=1`. Every unsupported
capability, layout or shape retains the Wave 12 B-stage kernel. Both production
arms retain Wave 20 fused compensated-MMA attention and the Wave 22 decision to
leave fused SwiGLU disabled.

## Gates, in order

1. Reconstruct the SHA-verified Wave 3 through Wave 22 stack and apply one
   incremental Wave 23 patch.
2. Pass the 64 historical `glcuda` library tests plus the new Wave 23 dispatch
   test. Host structural tests pin the 38,912-byte layout, four-block staging
   loop, two barriers per superstage and unchanged MMA/dequant sequence.
3. Assemble the full sm_75 module with `ptxas -v`. The retained and candidate
   entries must have zero stack, zero spill stores and zero spill loads.
4. Ask the CUDA driver for occupancy at both legal launches: 256 and 512
   threads, zero dynamic shared memory. At least one active block per SM is
   required at each geometry.
5. Require bit-exact f32 output versus the retained B-stage kernel for
   `ffn_gate_up` (9728x896), `ffn_down` (896x4864), and a ragged K32-tail
   diagnostic. Direct timing is a screen, not a production verdict.
6. Run the complete CUDA parity suite serially with the candidate selected.
7. Run four position-balanced production pairs in orders A/B, B/A, B/A, A/B,
   with 5 cold, 5 warmup and 10 measured iterations per arm. Every session
   must match the glproc token oracle exactly, 50/50, and the dispatch audit
   must observe K128 only in the candidate arm.
8. Retain only if median production prefill improves by at least 5%, every
   pair is positive, every paired decode delta is at least -5%, and median
   session-max latency regresses by at most 5%.

A retention result requires a second run of byte-identical notebook content.
Any compiler, resource, parity, oracle or hard-regression failure terminates
the wave without fix-forward. A harness-only failure may be corrected only
while the embedded kernel patch remains byte-identical.
