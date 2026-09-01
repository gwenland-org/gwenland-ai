# glcuda Wave 25 - 16-byte cooperative-stage design gate

**Date:** 2026-09-01
**Target:** Tesla T4 (`sm_75`), Qwen2.5-0.5B-Instruct Q4_K_M
**Status:** frozen before implementation
**Parent:** Wave 24 rejected independent MMA accumulator chains at -0.65%

## One question

The retained Wave 12 B-stage kernel moves each 32-byte activation or weight
payload with four lanes issuing one 8-byte global load and one 8-byte shared
store each. Wave 25 asks whether using two lanes and one 16-byte vector load
plus one 16-byte vector store per lane reduces memory-instruction issue cost
without changing the data image or occupancy tier.

This is a copy-width experiment. It is not another prefetch, larger tile,
superstage, accumulator-chain, or weight-format experiment. Those mechanisms
were already measured in Waves 5, 6, 14, 17, 23, and 24.

## Candidate and invariants

The candidate entry is `gl_gemm_mma_q8_bstage_v16`, selected only by
`GLCUDA_GEMM_V16=1`. For both 32-byte payloads in every K32 iteration:

- activation and B-stage quant bytes change from four lanes x 8 bytes to two
  lanes x 16 bytes;
- the row mapping changes from `tid >> 2` / `(tid & 3) * 8` to
  `tid >> 1` / `(tid & 1) * 16`;
- inactive loader lanes are predicated before either global or shared access;
- already-dead address temporaries should be reused for the second 64-bit
  vector element where possible; and
- activation f32 scales and weight f16 scales retain their existing scalar
  staging paths.

The candidate must preserve byte-for-byte semantics of the retained path:

- the same K32-major prepacked weight image;
- N64 at 256 threads and N128 at 512 threads;
- the same M64 grid and launch-selection rule;
- exactly 9,728 bytes of static shared memory with the same 48-byte row pitch;
- two barriers per K32 iteration and 16 static `m8n8k16` instructions;
- identical shared-memory bytes at the compute barrier;
- identical integer dot product, K32 dequantisation, f32 FMA order, guards,
  vector output stores, and ragged-tail behavior; and
- Wave 20 `mma4-fused` attention in both production arms.

Rejected Wave 22 fused-SwiGLU, Wave 23 K128, and Wave 24 MMA2 paths stay off.
The retained entry and every unrelated kernel must remain unchanged.

## Immutable compiler and correctness gates

The uploaded notebook reconstructs the SHA-verified Wave 3 through Wave 24
stack at base revision `3bce8dd7b8aaa2765855ab927c611b54981f9241`, then applies one
Wave 25 patch whose SHA-256 is recorded before upload.

Before any production timing is allowed, all of the following must pass:

1. every embedded historical patch and the Wave 25 patch matches its recorded
   SHA-256, applies cleanly, and leaves `git diff --check` clean;
2. the runtime is a Tesla T4 with compute capability 7.5;
3. structural assertions find the separate candidate entry, exactly two
   vector activation staging instructions, exactly two vector B staging
   instructions, the unchanged scalar-scale paths, 16 MMA instructions, two
   barriers, and the unchanged 9,728-byte shared image;
4. `ptxas -v -arch=sm_75` finds retained and candidate entries; the candidate
   uses at most 52 registers, exactly 9,728 bytes of static shared memory, and
   zero stack, spill stores, or spill loads;
5. the driver reports at least four active blocks/SM at 256 threads and two at
   512 threads for the candidate;
6. the reconstructed stack passes exactly 66 `glcuda` library tests;
7. serial release CUDA parity passes all 33 tests with `GLCUDA_GEMM_V16=1`;
   the retained Wave 20 attention flag is deliberately absent from this test
   process because several parity tests assert the unconfigured default;
8. direct retained/candidate records cover production `ffn_gate_up`
   9728x896x244, production `ffn_down` 896x4864x244, and a ragged N64 K160
   diagnostic; all timings are finite and positive and every output is bit
   exact; and
9. for **each** production direct-screen shape, `retained_us / candidate_us`
   is at least 0.95. Failure of either shape is an immediate hard stop.

A compiler, resource, structural, test, parity, exactness, or direct-screen
failure stops the wave. There is no fix-forward after this gate is frozen. A
harness-only repair may create a new notebook version only when the embedded
Wave 25 kernel patch remains byte-identical; both versions and the reason must
be recorded.

## Immutable production protocol

One release `glbench` binary runs both arms against the pinned GGUF SHA-256
`74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`.
The fixed prompt must tokenize to 244 tokens. Each session uses prefill,
`max_new_tokens=1`, seed 42, temperature 0, and `--verify-against glproc`.

Before each arm is constructed, the harness removes every historical GEMM
experiment variable, including `GLCUDA_GEMM_K128`, `GLCUDA_GEMM_MMA2`, and
`GLCUDA_GEMM_V16`. The common environment is exactly the retained production
stack: force-Q8, grid2d, exact Q8 glue fusion, N128 selection, B-stage GEMM,
and Wave 20 compensated-MMA attention. Only the candidate arm adds
`GLCUDA_GEMM_V16=1`.

After one untimed stabilization launch per arm, four position-balanced pairs
run in orders A/B, B/A, B/A, A/B. Every arm/session contains 5 cold, 5 warmup,
and 10 measured iterations. The harness requires:

- the `glcuda`/CUDA engine contract in all sessions;
- `mma4-fused` attention in both arms and `bstage-v16` dispatch only in B;
- exactly 5 cold and 10 measured positive timing records per session;
- exact 50/50 token-oracle agreement in all eight sessions; and
- no unregistered environment difference between arms.

P50, P90, P99, MAD, session maximum, cold latency, and decode throughput are
reported descriptively. For continuity, the primary throughput decision is
the ratio of the medians of the four session-P50 values. The four within-pair
deltas and their median are also reported explicitly.

## Decision

`RETAIN` requires all of these simultaneously:

- median prefill throughput improvement at least +5%;
- every one of the four paired prefill deltas positive;
- every paired decode delta at least -5%;
- median session-maximum latency delta no worse than +5%; and
- every correctness, dispatch, and oracle gate above passes.

An all-positive +2% to less than +5% result is measurable but remains opt-in.
Anything else keeps the retained K32 B-stage path as production default. A
`RETAIN` verdict requires a second Kaggle run of the byte-identical notebook
SHA; a rejection does not.

The executed notebook, patch and notebook hashes, compiler/resource logs,
driver occupancy, direct records, all eight `glbench` sessions, production
report, and failure archive (when applicable) are evidence, not scratch data.
