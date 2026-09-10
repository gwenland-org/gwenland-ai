# Wave 60 - N32/M32 production gate

## Outcome

Wave 60 repaired the context-sensitive Wave 59 overlay, but the T4 run stopped
at the host-test gate before the candidate PTX was assembled. There is still no
GPU correctness, resource, timing, or production-retention verdict for
`gl_gemm_mma_q8_bstage_n32_m32`.

## Intended deliverable

The run was designed to execute these gates in order:

1. reconstruct retained Wave 50 production;
2. install complete candidate files with SHA-256 verification;
3. require `ptxas` resources, driver occupancy, and bit-exact device parity;
4. require at least 1.10x direct speedup on QKV and FFN-down;
5. only then run ten balanced retained-N16 versus candidate-N32 production
   `glbench` pairs with the pinned Q8 model and exact 50/50 token oracle.

No later gate ran after the host-test failure.

## What the repair proved

- Kernel: `jinxsuperdev/glcuda-t4-wave60-n32-production`, version 1.
- Device: Tesla T4, compute capability 7.5, driver 580.159.04.
- The five candidate files were written byte-for-byte from their embedded
  snapshots and every SHA-256 matched.
- The reconstructed tree passed `git diff --check`.
- The Wave 59 overlay-context failure did not recur.

## Failure

The remote `glcuda` library suite reported:

```text
test result: FAILED. 62 passed; 2 failed; 0 ignored; 0 measured; 0 filtered out; finished in 2.87s
```

The two failures were:

```text
kernels::tests::sm75_ptx_is_structurally_sound
sm_75 PTX is missing gl_gemm_mma_q8_bstage_pipe(

tests::capabilities_match_hardware_reality
driver reported available, init must succeed:
Engine("cuModuleGetFunction failed: CUDA_ERROR_NOT_FOUND")
```

The complete-file strategy was correct, but its closure was incomplete. The
current `glcuda/src/kernels/mod.rs` snapshot expects the retained production
entry `gl_gemm_mma_q8_bstage_pipe`; the reconstructed historical
`glcuda_sm75.ptx` predates that entry. Wave 60 embedded the new N32 module but
not the current retained SM75 module.

This is a harness composition failure. It is not evidence that the N32 PTX
assembles, fails, matches, corrupts output, wins, or loses.

## Next gate

Wave 61 should add `glcuda/src/kernels/glcuda_sm75.ptx` to the hash-pinned
snapshot closure, with current SHA-256
`2a87248b17bbd3aef50209295e1a4c2f71d9fa559f14cb84564a99fd7f583e55`,
and rerun the same ordered gates without changing the candidate kernel.

The failed evidence archive is
`benchmarks/glcuda-t4-wave60-n32-v1-failed.zip` with SHA-256
`ee0a2d8fe758813e975dba1125d861ca77d29ab938496aa3a203c9c4864bea15`.
