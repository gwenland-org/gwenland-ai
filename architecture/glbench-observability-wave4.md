# glbench CUDA Launch Distributions - Wave 4

Date: 2026-09-24

## Goal

Preserve the per-launch timing evidence behind Wave 2's aggregate CUDA hotspot
table. A total and launch count reveal average cost, but hide tail latency and
run-to-run spread inside one kernel family.

## Raw evidence contract

Each `telemetry.launches.entries[]` record stores `samples_ms`: one CUDA-event
duration for every successfully timed launch in observation order. Collection
keeps Wave 2's global limit of 16,384 event pairs, so raw timing retention and
archive growth remain bounded.

`total_ms` and `launches` stay in schema v2 for compatibility. When raw samples
exist, archive readers recompute both values from the samples instead of
trusting the duplicated derived fields. Legacy entries without `samples_ms`
continue to use their archived aggregate values.

## Distribution contract

Terminal and Markdown reports derive P50, P90, and P99 from raw samples using
linear interpolation between adjacent ranks (type 7, matching glbench's
existing session statistics). A legacy entry renders `-` for percentiles; it
does not manufacture a distribution from a mean.

Samples are grouped by dispatch kind, entry name, and Wave 3 launch config.
Two launches of the same PTX entry with different grid, block, shared-memory,
or compiled-resource records therefore remain separate distributions.

## Interpretation boundary

The native observer starts at CUDA engine initialization, so this wave's raw
samples include warmup and measured benchmark work. The report and coverage
text must not describe them as measured-window-only samples. Separating warmup
from the authoritative measurement window requires an explicit observer reset
or epoch contract in a later wave.

The samples remain diagnostic GPU work durations. Enabling telemetry makes the
session instrumented, and overlapping streams mean summed durations are not
end-to-end wall time.

## Gate

Host-side tests must cover percentile math, raw-sample archive round-trip,
recomputation of duplicated aggregates, legacy archives with no samples, and
terminal/Markdown tail rendering. A T4 run is required before interpreting any
real distribution; this implementation wave makes no performance claim.
