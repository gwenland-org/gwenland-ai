# Gwen Changes - glbench Observability Wave 4

Date: 2026-09-24

## Problem

The native CUDA launch profile retained every timing event but discarded the
individual durations after computing total time and launch count. That made a
stable kernel and a tail-heavy kernel look identical when their means matched.

## Fix

- Preserve every successfully timed dispatch duration as bounded raw evidence.
- Group samples by dispatch identity and exact Wave 3 launch configuration.
- Recompute aggregate time and count from raw samples when reading archives.
- Derive P50, P90, and P99 with the same interpolation convention as existing
  glbench statistics.
- Render unavailable percentiles as `-` for legacy aggregate-only archives.

## Limits

The observer still covers the CUDA engine lifetime, including warmup. These
distributions are therefore diagnostic and instrumented, not authoritative
production-window latency. No T4 timing or speed claim is part of this host
gate.
