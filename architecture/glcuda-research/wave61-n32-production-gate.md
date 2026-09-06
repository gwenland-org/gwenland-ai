# Wave 61 - N32/M32 production gate

## Outcome

Wave 61 completed the production source snapshot closure and passed every
retained-code validation gate, but stopped before assembling the candidate
because the notebook did not export its discovered `ptxas` path across phases.
There is still no candidate resource, device-parity, timing, or production
verdict.

## Gates passed on Tesla T4

- All six snapshot files were written with matching SHA-256 values.
- `git diff --check` passed.
- `glcuda` host suite: 64 passed, 0 failed.
- The retained SM75 module assembled and initialized through the CUDA driver.
- Full CUDA parity: 33 passed, 0 failed.
- The retained register-Q direct control remained bit-exact and measured
  1.2942x in its existing diagnostic.
- The release `glbench` binary built successfully.

These results close the incomplete-source problem found in Wave 60.

## Failure

The candidate screen stopped on this exception:

```text
NameError: name 'PTXAS' is not defined
```

The bootstrap's `compile_phase()` found `ptxas` and assigned it to a local
variable. The candidate screen is a later top-level phase and referenced that
name without it having been exported to notebook-global scope. Consequently,
`ptxas-wave59-v.log` was never created and the candidate was not assembled.

This failure says nothing about the N32 kernel's resources, correctness, or
speed. Its PTX remained byte-identical to Waves 59 and 60, SHA-256
`0f3a390c1bc9509836aed1ccd450dd341b5f6d91cf5c5c18916f0b7913ca86f6`.

## Next gate

Wave 62 should export the resolved `ptxas` path from `compile_phase()` or
resolve it independently in the candidate screen, then rerun the same ordered
candidate and ten-pair production gates without modifying the kernel.

Evidence is archived at `benchmarks/glcuda-t4-wave61-n32-v1-failed.zip`,
SHA-256
`ca6f4c8c9cbf6988488d7694ee027106dfdba295512285a1584a49aea73d3dda`.
