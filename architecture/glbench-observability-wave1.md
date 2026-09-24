# glbench Observability MVP - Wave 1

Date: 2026-09-24

## Goal

Make bottleneck evidence portable and stop instrumented throughput from being
mistaken for production performance. This wave changes truth plumbing only. It
does not add CUDA kernel counters, NVML sampling, or Nsight import.

## Delivered contract

1. A default inference run is `production` and no longer enables glproc stage
   profiling implicitly.
2. `--profile stages` starts a child process with the selected engine's
   profiling variable set before runtime threads exist. The resulting session
   is `instrumented`, and terminal/Markdown/comparison output labels its tok/s
   non-authoritative.
3. Existing engine profiling variables are detected. A caller cannot obtain a
   `production` label while `GLPROC_PROFILE`, `GLCUDA_TELEMETRY`, or either
   glcuda phase profiler is active.
4. Telemetry round-trips from JSON archives: prefill/decode stages, raw byte and
   MAC counts, backend/kernel selection, memory, and MoE routing.
5. Each adapter-backed run records a dispatch-configuration fingerprint and
   the whitelisted overrides that produced it. No unrelated environment value
   is archived.
6. A legacy archive with no measurement-mode field reads as `unknown`, never
   as production.

## Schema decision

The archive remains schema v2. The new metadata and workload fields are
additive, and the reader has explicit meanings for every missing field. No v2
field was removed or changed. Derived telemetry numbers are not trusted on
read; consumers recompute them from raw timings, calls, bytes, and MACs.

## Known limits carried to Wave 2

- The fingerprint covers engine configuration inputs, not the final resolved
  hardware-dependent kernel suite. Backend telemetry remains the source for
  actual selected paths.
- glcuda currently exposes a prefill stage split but not a complete native
  per-kernel launch stream or occupancy/counter record.
- Behavior reports still do not round-trip. They are separate derived data and
  are outside this truth-plumbing wave.
- Source revision and model content hashing are not captured yet. Model hashing
  must avoid warming the model file before the production measurement.

## Gate

The code gate requires the pre-change and post-change glbench test suites, a
focused telemetry archive round-trip test, instrumented-report authority tests,
formatting, and a clean diff check. This wave makes no performance claim.
