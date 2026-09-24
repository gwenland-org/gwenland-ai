# Wave 128: register-reuse T4 device gate

## Deliverable

Wave 128 validates the Wave 126 `%rP_next` register reuse and Wave 127 launcher
wiring on a Tesla T4. It is a non-production device gate; no throughput claim
or retention decision may be made from this wave.

## Ordered gates

1. Checkout the latest remote-available base `5de5be3`, then apply one full
   SHA-verified patch that reconstructs local commit `f979647` plus the exact
   dirty Wave126/127 candidate. No push is required for the device run.
2. Run 67 host library tests on Linux.
3. Assemble `glcuda_sm75_wave88.ptx` for `sm_75` with `ptxas -v`. The fused
   entry must have at most 80 registers, one barrier, exactly 16,384 bytes
   static shared memory, and zero stack, spill stores, and spill loads.
4. Load the module through the CUDA driver and report at least three resident
   256-thread CTAs per SM.
5. Run the CUDA parity suite serially with `--test-threads=1`.
6. Run the direct Wave126 harness. Retained and fused Q8 bytes plus f32 scale
   bits must match exactly for the 244-row production shape, including three
   full 64-row slabs and its 52-row ragged tail.

Any build, resource, module-load, occupancy, parity, or exact-output failure
stops the wave. Standalone kernel timing is diagnostic only and cannot retain
the candidate.

If every gate passes, Wave 129 may perform the counterbalanced production
wall-time A/B required to judge the 15,000 prefill tok/s goal.
