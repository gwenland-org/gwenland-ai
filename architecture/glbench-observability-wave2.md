# glbench Native CUDA Launch Profile - Wave 2

Date: 2026-09-24

## Goal

Expose which CUDA device entries consume GPU time without presenting an
instrumented run as production throughput. This wave changes observation only:
no PTX, launch geometry, kernel selection, or model arithmetic changed.

## Collection contract

`GLCUDA_TELEMETRY=1` must be present before CUDA initialization. The driver
then brackets each successful direct kernel launch with CUDA events on the
same stream. CUDA Graph capture-time kernel nodes are deliberately skipped;
each graph replay is timed as one `graph_replay` dispatch because the driver
launch seam cannot truthfully attribute its internal nodes.

The snapshot reports:

- entry name and kind (`kernel` or `graph_replay`);
- launch count, accumulated device milliseconds, milliseconds per launch, and
  share of summed GPU work;
- observed, timed, and untimed dispatch counts;
- timing source and the exact coverage boundary.

Summed event durations are work attribution, not wall time. Independent CUDA
streams may overlap. The coverage also begins at engine initialization, so it
includes warmup and measured benchmark iterations. Collection is capped at
16,384 event pairs; later dispatches remain counted as observed but untimed.

## Disabled path

Without `GLCUDA_TELEMETRY`, CUDA allocates no profiling events, acquires no
profiler lock, and stores no records. `Cuda::launch` takes one predictable
`Option::is_some` guard before the retained raw launch call. A device A/B is
still required before claiming that guard is performance-neutral.

## Archive contract

The additive `telemetry.launches` object carries raw entry timings and launch
counts. glbench recomputes summed work and shares from those raw fields. Older
schema-v2 archives omit the object and continue to read as `None`.

## Gate

Host-side tests must cover the shared launch-profile arithmetic, full archive
round-trip, explicit graph-replay rendering, and old archives with no launch
profile. CUDA parity remains unchanged because no kernel code changed. Real
event timing and disabled-path overhead still need a T4 device run; this wave
makes no production-speed claim.
