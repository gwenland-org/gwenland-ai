# Gwen Changes - glbench Observability Wave 5

Date: 2026-09-24

## Problem

CUDA launch distributions included cold-start and warmup submissions even
though glbench's headline statistics exclude those phases. The samples were
real, but they did not describe the same observation window as throughput.

## Fix

- Add an engine-level telemetry-window boundary with a safe default no-op.
- Open the window after warmup and before every authoritative measured clock.
- Clear glcuda launch samples, counters, cached aggregation, and event budget.
- Preserve immutable kernel-resource metadata across the reset.
- Archive coverage text that states whether an explicit reset occurred.

## Limits

The launch profile is still diagnostic and instrumented. CUDA events can
perturb execution, overlapping streams prevent sums from being wall time, and
real distributions still require a T4 validation run before interpretation.
