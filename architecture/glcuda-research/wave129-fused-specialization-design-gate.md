# Wave 129: fused-only specialization compiler gate

## Frozen hypothesis

Wave 128 proved that renaming one dead PTX temporary cannot remove the fused
N16-prefetch spill. The prior candidate still compiled one entry containing
both the retained f32 epilogue and the fused SwiGLU-to-Q8 epilogue, selected by
a runtime predicate. That mode state and the mutually exclusive epilogues can
extend compiler liveness across the final MMA boundary.

Wave 129 tests one compiler-only change: specialize the candidate as a
fused-only entry. The retained Wave 88 entry remains untouched in its own PTX
image. The fused arithmetic, instruction order, accumulator layout, K32
prefetch schedule, 256-thread launch, 16 KiB aliased shared image, and Q8
quantization order are held fixed.

## Deliverable and stop rule

The wave delivers an isolated PTX entry plus a structural host contract and a
Tesla T4 `ptxas -v` result. It does not wire production dispatch and therefore
cannot claim throughput.

The compiler gate requires:

- at most 80 registers per thread;
- exactly one barrier resource and 16,384 bytes static shared memory;
- zero stack frame, spill stores, and spill loads.

Any compiler-resource failure stops the wave before driver load, occupancy,
parity, integration, or timing. Passing only authorizes a later device-parity
wave; it does not retain the candidate.

## Host evidence

- baseline before the edit: 67 passed, 0 failed;
- structural candidate test added: expected suite count 68;
- candidate PTX SHA-256:
  `80cdd36f64ce8873679d1a75ea0e01ad7525177a26f4ea69fe0d37737be4d06b`.

The repository-wide format check has pre-existing failures outside this diff
in `glbench`, `glictus-caliburni`, and `packages/mcp`. The touched Rust file is
checked independently with `rustfmt --check`; those unrelated files are not
rewritten in this wave.
