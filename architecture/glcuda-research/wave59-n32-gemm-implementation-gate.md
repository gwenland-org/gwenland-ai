# Wave 59 - N32/M32 GEMM implementation gate

## Outcome

Wave 59 implemented an isolated, opt-in SM75 Q8 GEMM candidate, but produced
no GPU verdict. The first T4 submission stopped before `ptxas` because the
notebook overlay did not apply to its historical reconstruction. No
performance or resource number is claimed from this run.

## Candidate

- Entry: `gl_gemm_mma_q8_bstage_n32_m32`
- Activation tile: M32
- Output tile: N128 per 128-thread CTA
- Warp tile: M32 x N32
- Static source budget: 8,064 bytes shared memory and `.maxnreg 80`
- Dispatch: `GLCUDA_GEMM_N32=1`, only where the retained N16 dispatcher would
  select its narrow-grid M32 arm
- Production default: unchanged

The retained narrow CTA covers M64 x N64 with eight warps. The candidate covers
the same 4,096 output elements as M32 x N128 with four warps. Each warp reuses
one A fragment across four adjacent N8 outputs. Per K32 block this reduces the
static shared-to-register load sites from 40 to 24 per CTA while preserving 64
MMA operations and the existing K32 dequantization order.

## Local gates

- Baseline before edits: `glcuda` 63/63, `glcore` 133/133, `glproc` 104/104
  with one ignored test; all integration suites passed.
- After edits: `glcuda` 64/64.
- The dedicated device test and direct benchmark both compile.
- This Windows host has no CUDA driver or `ptxas`, so the device test correctly
  took its explicit skip path. It is not GPU parity evidence.
- Structural contract: 32 `mma.sync`, four A `ldmatrix.x2`, two B
  `ldmatrix.x4`, two barriers, 16 vector stores, ASCII with LF line endings.

## T4 submission v1

- Kaggle kernel: `jinxsuperdev/glcuda-t4-wave59-n32-gemm`, version 1
- Device allocated: Tesla T4, compute capability 7.5, driver 580.159.04
- Stop phase: `compile-resource-correctness`
- Error: `Wave 59 overlay apply failed`

The archived Wave50 notebook reconstruction still contains the rejected
Wave21-Wave24 experimental module declarations. The local production snapshot
removed those declarations. The Wave59 patch was generated against production,
so its first `mod.rs` context did not match the historical reconstruction.
`runner.rs` and all three new files passed the dry-run patch check; the failure
was confined to that stale `mod.rs` context.

## Decision and next gate

Status is **no GPU verdict**, not reject. Wave 60 must construct its overlay
against the actual production commit snapshot (or ship complete candidate
files with hash verification), then run these gates in order:

1. `ptxas -v`: at most 80 registers, exactly 8,064 bytes static shared memory,
   zero stack and zero spills.
2. T4 driver occupancy: at least six active 128-thread CTAs per SM.
3. Dedicated device parity: bit exact to retained N16/M32 on ragged and narrow
   shapes.
4. Interleaved direct screen: at least 1.10x on both QKV and FFN-down shapes.
5. Only after those pass, a separate production `glbench` A/B may make a
   retention claim.

The failed evidence archive is
`benchmarks/glcuda-t4-wave59-n32-v1-failed.zip` with SHA-256
`5ab48adf9c742c32d270480b6143bb056b0954ffaff87cf9c9931af1a704214d`.
