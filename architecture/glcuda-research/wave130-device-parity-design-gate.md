# Wave 130: fused specialization device-parity gate

## Deliverable

Wave 130 loads the zero-spill Wave 129 PTX as an isolated, opt-in CUDA module
and exposes one direct launch wrapper. The production runner remains untouched;
the new entry cannot be selected by normal inference in this wave.

The candidate is enabled only when the retained Wave 123/Wave 88 flags and
`GLCUDA_N16_FUSED_SWIGLU=1` are present. Every unsupported environment or shape
keeps the existing modules and dispatch unchanged.

## Ordered gates

1. Existing host tests plus the new shape/ABI structural checks pass.
2. On a Tesla T4, `ptxas -v` reproduces Wave 129's zero-stack, zero-spill,
   <=80-register, one-barrier, 16,384-byte shared image.
3. CUDA driver module load and symbol lookup succeed.
4. Driver occupancy is at least three resident 256-thread CTAs per SM.
5. The serial CUDA parity suite passes with `--test-threads=1` while the
   candidate module is loaded.
6. The direct production-shape harness compares the retained stacked
   N16-prefetch GEMM plus no-store SwiGLU against the specialized entry. All Q8
   bytes and f32 scale bits must match for 244 rows, including three complete
   64-row slabs and the 52-row ragged tail.

Any failure stops the wave. Direct kernel timing is diagnostic only. Production
dispatch and the 15,000 prefill tok/s retention decision require a later
counterbalanced production wall-time wave.
