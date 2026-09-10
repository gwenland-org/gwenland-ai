# Wave 62 - N32/M32 production gate

## Outcome

Wave 62 produced the first valid device verdict for the isolated N32/M32 Q8
GEMM candidate. The kernel assembled at its exact resource budget and was
bit-exact, but lost decisively to the retained N16/M32 kernel on every tested
shape. It was rejected before production `glbench`, as required by the direct
hard stop.

## Environment and method

- Kaggle Tesla T4, compute capability 7.5, driver 580.159.04.
- Candidate PTX SHA-256:
  `0f3a390c1bc9509836aed1ccd450dd341b5f6d91cf5c5c18916f0b7913ca86f6`.
- Ten warmups, 100 measured launches, five interleaved repeats per shape.
- Retained and candidate operated on the same allocations and inputs.
- Candidate output was compared bit-for-bit against retained N16/M32.

## Gates passed

- Six-file production snapshot: all SHA-256 values matched.
- `git diff --check`: passed.
- `glcuda` host suite: 64 passed, 0 failed.
- Full CUDA parity: 33 passed, 0 failed.
- Candidate device parity: 1 passed, 0 failed, bit-exact.
- `ptxas`: 80 registers, 8,064 bytes static shared memory, one barrier, zero
  stack, zero spill stores, and zero spill loads.
- Driver occupancy: candidate 6 active 128-thread CTAs/SM versus retained 4
  active 256-thread CTAs/SM.

## Direct result

| Shape | Retained N16/M32 | Candidate N32/M32 | Speedup | Result |
|---|---:|---:|---:|---|
| QKV 1152x896x244 | 55.870 us | 63.865 us | 0.8748x | -12.5% |
| FFN-down 896x4864x244 | 178.834 us | 216.932 us | 0.8244x | -17.6% |
| Ragged 136x160x17 | 5.136 us | 6.989 us | 0.7349x | -26.5% |

The required direct gate was at least 1.10x on both production shapes. Neither
shape was close, so the production A/B was correctly not executed.

## Interpretation

Higher block residency was not the limiting variable. The candidate reduced
the CTA from eight warps to four while increasing per-thread registers from the
retained M32 kernel's measured 64 to 80. Six resident candidate CTAs provide 24
resident warps, below the retained path's 32 resident warps from four CTAs. The
N32 tile therefore raised the blocks-per-SM number while reducing the resident
warp pool and instruction-level latency hiding that matters to this kernel.

This closes the N32/M32 direction. It must not be promoted or production-tested
through a weaker gate.

## Deviation and next risk

The planned ten-pair production A/B did not run because the direct hard stop
failed. The 15,000 tok/s production target remains open; Wave 62 provides no new
production throughput number.

The next wave should remove the rejected opt-in N32 implementation and choose a
different mechanism with enough end-to-end leverage. Repackaging this geometry
or pursuing occupancy alone is not supported by the measurements.

Evidence is archived at `benchmarks/glcuda-t4-wave62-n32-v1-failed.zip`,
SHA-256
`eea3bcc5195550595378713259e9d0420e4ae44fb8bfb23bae2071e431c1837f`.
