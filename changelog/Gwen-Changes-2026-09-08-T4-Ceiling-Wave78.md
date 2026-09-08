# T4 Ceiling Wave 78 - fused compensated-MMA AV candidate

## Problem

Wave 77 proved twice on Tesla T4 that the retained row-major V cache can feed
four-product compensated-f16 MMA AV more than 4x faster in isolation. The
production attention kernel still accumulated AV scalarly, and directly
replacing that loop was impossible because its sixteen query rows normalized
their probabilities independently across four warps.

## Change

- Added an opt-in sm_75 fused-attention entry that preserves Wave 48 QK and
  softmax, normalizes probabilities in shared memory, then computes AV with
  four cooperative MMA warps.
- Preserved the row-major KV cache and reused the existing score allocation;
  no hot-path allocation or second V image was added.
- Kept `mma4-regq` unchanged as the A/B baseline and fallback. The candidate
  requires `GLCUDA_ATTN_MMA4_AV=1` on top of the existing MMA4 flags.
- Added auditable dispatcher telemetry, structural PTX contracts, production
  plus ragged/tail GPU parity coverage, and a dedicated interleaved direct A/B
  harness.

## Evidence and status

The glcuda host suite passes 64/64, the serial parity target compiles and passes
33/33 on the GPU-less Windows host, and the release direct harness builds. The
PTX is ASCII/LF-only and structurally contains the intended 32 QK plus eight AV
MMA sites and four barriers.

This is not a performance result. T4 `ptxas`, device parity and direct timing
remain the next gate; production `glbench` follows only if those pass.
