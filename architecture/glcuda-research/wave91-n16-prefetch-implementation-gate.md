# Wave 91 - N16 register-prefetch implementation gate

## Outcome

Wave 91 implements the Wave 87 candidate behind
`GLCUDA_GEMM_N16_PREFETCH=1`. The new entry
`gl_gemm_mma_q8_bstage_n16_prefetch` lives in an isolated SM75 PTX module,
which is loaded only when the experiment is explicitly enabled. The retained
SM75 image and default dispatch remain unchanged.

The candidate is selected only for `out=4864, in=896, ntok=244`, the two FFN
gate/up projections in the pinned Qwen2.5-0.5B production workload. QKV, down,
narrow M32, other token counts, and all decode paths retain their existing
entries.

## Implementation invariants

- The current K32 block is committed from prefetch registers to the retained
  9,728-byte shared image.
- The next block's two A loads, two B loads, activation scale, and weight scale
  are issued before the current N16 compute body.
- The retained eight `ldmatrix.x2` A loads, one `ldmatrix.x4` B load, 32
  explicit `m8n8k16` MMA instructions, f32 epilogue order, and two barriers
  remain intact.
- The final iteration predicates off its next-block loads.
- `.maxnreg 80` is a ceiling to be checked by PTXAS, not a measured resource
  claim.

## Local verification

`cargo test -p glcuda --lib --locked` passed 65/65 tests. The added structural
test checks the isolated entry, balanced PTX, instruction and barrier counts,
shared allocations, register ceiling, prefetch registers, and exact production
shape guard.

Workspace-wide and package-wide rustfmt checks remain unusable as gates because
unrelated pre-existing files differ from the installed formatter. `git diff
--check` passed, and the compiler/test suite is authoritative for this change.

## Remaining device gate

No CUDA performance or correctness claim is made in this wave. The next wave
must package a self-contained T4 harness that:

1. compiles retained and candidate entries with `ptxas -v`;
2. records registers, shared memory, spills, and driver occupancy;
3. compares output bit-for-bit at 4864x896x244;
4. runs two independent counterbalanced direct A/B measurements;
5. stops before production unless both candidate/retained medians are at least
   1.10x.

The fastest measured experimental production result remains 11,377.9 tok/s;
15,000 tok/s is not yet proven.
