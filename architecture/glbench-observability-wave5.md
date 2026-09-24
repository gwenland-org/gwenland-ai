# glbench Measured Telemetry Window - Wave 5

Date: 2026-09-24

## Goal

Align accumulated engine telemetry with the same measured iterations that
produce glbench's throughput statistics. Wave 4 preserved CUDA launch samples,
but its engine-lifetime observer also retained cold-start and warmup work.

## Boundary contract

`GlEngine::begin_telemetry_window` marks a quiescent observation boundary.
glbench calls it after cold-start and warmup inference have returned, and
before starting the measured wall, CPU, or energy clocks.

The default implementation is a no-op for engines whose telemetry is static or
already represents only the most recent run. An engine that accumulates dynamic
samples must discard the old samples and counters at this boundary while
retaining immutable metadata caches.

## CUDA implementation

glcuda clears recorded event pairs, observed-launch counts, any cached
aggregation, and the 16,384-pair reservation count. It retains the launch
resource cache because function attributes and occupancy projections remain
valid for the loaded function and launch shape.

Coverage text distinguishes an engine-lifetime snapshot from a snapshot taken
after an explicit reset. It therefore never claims warmup exclusion merely
because a consumer happens to expect it.

## Timing boundary

The reset happens before glbench starts its authoritative measured clocks.
Destroying old diagnostic events therefore cannot inflate measured wall time,
CPU utilization, or energy. The telemetry snapshot is still taken before the
extra behavior-tracing run, so that run cannot enter the launch distribution.

## Interpretation boundary

The resulting CUDA launch distributions cover measured iterations only when
the engine is driven through this glbench phase sequence. They remain
instrumented GPU work durations, not production wall time, and overlapping
streams still make summed duration different from elapsed time.

## Gate

Host-side tests must prove that a reset clears dynamic evidence, restores the
bounded event budget, preserves resource-cache entries, and labels coverage
truthfully. Full glcore, glcuda, and glbench suites must remain green. A T4 run
is still required before interpreting real launch distributions; this wave
makes no performance claim.
