# Gwen Changes - glbench CUDA Observability

Date: 2026-09-24

## Problem

glbench could report end-to-end CUDA throughput and coarse stage totals, but
could not preserve enough evidence to identify a production launch bottleneck.
Early profiling also mixed diagnostic timing with production authority and
accumulated cold-start and warmup launches into the same distribution as the
measured iterations.

## Changes

- Separate default `production` sessions from opt-in `instrumented` profiling
  and label profiled throughput non-authoritative.
- Archive dispatch-configuration provenance and raw engine telemetry so
  inspect/export sees the same facts as the live report.
- Time successful direct kernels and CUDA Graph replays with same-stream CUDA
  events, retaining observed/timed coverage and bounded raw launch samples.
- Group direct launches by entry and exact launch configuration, including
  grid, block, shared memory, compiler resources and projected SM residency.
- Derive mean, P50, P90 and P99 from raw samples; legacy aggregate-only
  archives remain readable and show unavailable distribution fields as `-`.
- Open a fresh telemetry window after cold-start and warmup but before the
  measured clocks. Dynamic samples and the event budget reset; immutable
  resource metadata remains cached.

## Measurement contract

The observer retains at most 16,384 CUDA event pairs. Dispatches beyond the
cap remain observed but untimed. CUDA Graph replay is one opaque entry because
the replay seam does not expose internal kernel nodes. Summed GPU durations are
work attribution rather than wall time when streams overlap, and projected
residency is not achieved occupancy.

Instrumented results locate cost; they do not decide performance retention.
Production throughput still requires an uninstrumented glbench run, while real
launch distributions and driver-resource values require validation on the
target CUDA device.

## Compatibility

All additions remain optional schema-v2 fields. Missing measurement authority
reads as `unknown`; missing launch samples or resources remain unavailable
instead of being guessed. When raw samples exist, archive readers recompute
launch count and total duration from them rather than trusting duplicated
derived values.
