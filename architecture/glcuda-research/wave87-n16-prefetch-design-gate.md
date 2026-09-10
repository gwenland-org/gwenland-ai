# Wave 87 - N16 register-prefetch FFN design gate

## Decision

Wave 87 freezes one exact-Q8 experiment for the next device sprint: issue the
next K32 block's global loads early in the retained N16 B-stage kernel, then
consume the current block's two N8 output fragments while those loads are in
flight. The candidate is limited to the production FFN gate/up shape
`out=4864, in=896, ntok=244`; the retained entry, narrow M32 entry, down
projection, QKV projections, packing, scales, and epilogue remain unchanged.

This is not a resurrection of the rejected Wave 17 kernel. Wave 17 applied
register prefetch to the old N8 warp tile and measured only 1.038x/1.033x. The
retained Wave 27 tile now gives each warp two adjacent N8 fragments, doubling
the independent MMA and f32 accumulation work available between issuing and
consuming the next global loads. That changed latency-hiding window has never
been measured.

## Why this is the next admissible mechanism

- Wave 58 attributes 31.6% of profiled prefill time to FFN gate/up.
- Wave 16 measured roughly 25-27% of the old GEMM in staging, while removing A
  loads alone was worth only 1-2%; a useful candidate must overlap the complete
  A, B, and scale stage rather than optimize one operand.
- N16 and `ldmatrix` are retained and production-proven. N32, M32 on wide
  grids, K128, independent MMA chains, fused SwiGLU, and r256 are already
  rejected and stay out of scope.
- The Q8_0 bytes, per-K32 scales, integer MMA order, f32 conversion, scale
  multiplication, and accumulation order can remain bit-identical.

## Frozen implementation

Create a separate SM75 entry derived from `gl_gemm_mma_q8_bstage_n16`.

1. Prologue-load K32 block zero exactly as retained.
2. For every non-final iteration, issue the next block's aligned A/B and scale
   global loads into dedicated registers before computing the current block.
3. Run the current block's unchanged `ldmatrix`, four MMA instructions per
   M8 tile, scale conversion, and f32 FMAs.
4. Synchronize after all warps finish reading the current shared image, commit
   the prefetched registers to that image, synchronize again, and advance.
5. Run the final block without an out-of-range prefetch.

The candidate keeps one 9,728-byte shared image. It does not add a second
shared stage: SM75 lacks asynchronous copy, and a second image alone would not
overlap same-warp instruction issue. The causal lever is the longer N16
instruction window between global-load issue and use.

## Resource and dispatch contract

- Separate opt-in entry and environment flag; retained dispatch is unchanged.
- Candidate dispatch only when `out_dim == 4864`, `in_dim == 896`, and
  `ntok == 244` through the existing N16 wide-grid geometry.
- At most 80 registers/thread, exactly 9,728 bytes static shared memory, zero
  stack, zero spills, and the same barrier count per K32 iteration.
- Driver occupancy must remain at least three active blocks/SM for the
  256-thread production launch. This is measured on T4, not inferred from the
  PTX declarations.

The 80-register ceiling preserves the same register-limited residency tier:
`80 * 256 * 3 = 61,440`, below T4's 65,536-register SM budget. PTXAS allocation
rounding and driver constraints remain authoritative.

## Stop gates

The next wave stops immediately on the first failure:

1. host structural tests and `ptxas -v` resource gate;
2. serial CUDA parity, including ragged rows and an exact bit comparison with
   retained N16;
3. two independent counterbalanced direct A/B runs at the production gate/up
   shape, each requiring candidate/retained at least 1.10x;
4. only after the direct gate passes, immutable 50/50 token oracle and balanced
   production `glbench` sessions;
5. production retention requires every paired delta positive and session-P50
   throughput at least 1.05x retained.

Direct timings are feasibility evidence only. The project throughput remains
11,377.9 tok/s on the fastest experimental production stack until a complete
production run proves otherwise; 15,000 tok/s remains unproven.

## Boundary

Wave 87 changes documentation only. Wave 88 may implement exactly this
candidate after confirmation. A failed resource, parity, or speed gate removes
the candidate without modifying the production default.
