# glbench CUDA Launch Resources - Wave 3

Date: 2026-09-24

## Goal

Extend Wave 2's kernel-time ranking with the launch shape and resource pressure
needed to interpret a hotspot. This wave observes the existing production
dispatch. It does not change PTX, launch geometry, kernel selection, or model
arithmetic.

## Collection contract

Resource collection is active only when `GLCUDA_TELEMETRY` enabled the native
launch observer before CUDA initialization. For each distinct direct-kernel
handle and launch shape, glcuda records:

- grid and block dimensions;
- registers per thread;
- static and dynamic shared-memory bytes;
- local bytes per thread;
- projected active blocks and warps per SM.

Compiler/JIT resources come from `cuFuncGetAttribute`. Resident blocks come
from `cuOccupancyMaxActiveBlocksPerMultiprocessor`, using the exact block size
and dynamic shared-memory request. The latter accounts for hardware allocation
rules that byte arithmetic cannot reproduce reliably.

These queries are cached per function handle and launch shape. A long profile
therefore pays one resource lookup per distinct variant, not per launch. A
missing symbol or rejected query leaves that field unavailable and never fails
inference.

## Interpretation boundary

Active blocks and warps are projected residency, not measured achieved
occupancy. They describe how much work could reside on an SM under resource
limits; they do not reveal scheduler utilization, cache traffic, instruction
mix, or stall reasons. Those still require CUPTI/Nsight or equivalent hardware
counters.

CUDA Graph replay remains one opaque `graph_replay` entry. Its internal nodes
have no launch-resource record because the driver replay seam exposes only the
graph executable, not each captured kernel launch.

## Archive contract

The additive `telemetry.launches.entries[].config` object carries raw launch
and resource fields. A missing object means "not recorded", preserving old
schema-v2 archives and graph-replay honesty. Terminal and Markdown output place
resources in a separate table so projected residency cannot be confused with
timed GPU work.

## Gate

Host-side tests must cover resource-cache reuse, archive round-trip, legacy
entries with no resource object, and terminal/Markdown interpretation labels.
The CUDA symbols remain runtime-loaded and optional. A T4 instrumented run is
still required to validate real driver values; this wave makes no speed or
achieved-occupancy claim.
