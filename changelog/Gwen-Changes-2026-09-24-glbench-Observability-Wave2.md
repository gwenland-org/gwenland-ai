# Gwen Changes - glbench Observability Wave 2

Date: 2026-09-24

## Problem

The CUDA backend exposed coarse prefill stages, but glbench could not answer
which concrete kernel entry dominated GPU work. CUDA Graph decode also made a
naive per-kernel claim wrong because replay hides its internal launches.

## Root cause

Kernel names stopped at `KernelSet`, while timing lived in the higher-level
prefill runner. No shared telemetry shape described a native device dispatch,
its timing mechanism, or its coverage blind spots.

## Fix

- Retain each resolved PTX entry name beside its copyable CUDA function handle.
- When `GLCUDA_TELEMETRY` is active before initialization, record one CUDA
  event pair around every successful direct launch on its actual stream.
- Exclude capture-time graph nodes and record each replay honestly as
  `cuda_graph_replay` with kind `graph_replay`.
- Aggregate device time and launch counts by `(kind, name)` and expose observed
  versus successfully timed dispatches. Event-pair retention is capped at
  16,384; later dispatches remain visible as untimed instead of growing memory
  without bound.
- Archive and render timing source, coverage, total work, per-launch cost, and
  work share in terminal and Markdown reports.
- Preserve old schema-v2 archives: a missing `telemetry.launches` remains
  "not measured", never a zero profile.

## Limits

The profile covers the engine lifetime, including warmup. Event durations on
concurrent streams can overlap, so their sum is not end-to-end wall time.
Graph internals require a later CUPTI/Nsight import path. No T4 timing was run
in this Windows host gate, so this change claims observability, not speed.

## Open-source readability follow-up

The profiler's internal comments now define "dispatch", explain event and
CUDA Graph lifetimes, show why event retention is bounded, and document every
diagnostic fallback. The target reader is a capable student encountering CUDA
events for the first time, while experienced contributors can still scan the
contracts without following each branch.
