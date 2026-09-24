# Gwen Changes - glbench Observability Wave 3

Date: 2026-09-24

## Problem

Wave 2 could rank CUDA entries by time, but a hotspot name alone did not show
whether its launch shape, register allocation, shared memory, local memory, or
projected residency deserved investigation.

## Fix

- Dynamically resolve the optional CUDA Driver function-resource query.
- Cache launch resources per kernel handle and exact launch shape while native
  telemetry is enabled.
- Record grid, block, registers/thread, static and dynamic shared memory,
  local bytes/thread, active blocks/SM, and active warps/SM.
- Keep graph replays resource-free instead of inventing values for hidden
  graph nodes.
- Round-trip the additive resource object through schema-v2 JSON archives and
  render it separately in terminal and Markdown reports.
- Keep legacy entries readable as resource-unavailable.

## Limits

Blocks/SM and warps/SM are CUDA's projected residency, not measured achieved
occupancy. Cache behavior, issue utilization, and stall reasons remain outside
glbench's native launch seam and require hardware-counter tooling. No T4 run or
performance claim is part of this host-side implementation gate.
