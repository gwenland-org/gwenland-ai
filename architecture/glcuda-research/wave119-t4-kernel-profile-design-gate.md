# Wave 119 - T4 kernel-profile design gate

## Objective

Locate the next production-sized prefill lever after Wave 118 without changing
the inference path. The current `glbench` telemetry is sufficient for an
eight-stage roofline, but not for choosing another kernel mechanism: it reports
only the last measured prefill iteration, combines several launches in the
elementwise buckets, and cannot split the fused attention kernel into QK,
softmax, and AV. It also has no instruction, scheduler-stall, or cache counters.

Wave 119 is observation-only. It profiles the exact Wave 118 Q8 dispatch on a
Tesla T4 and must not produce a retention or throughput claim from profiler
timings.

## Locked workload

- pinned Qwen2.5-0.5B-Instruct Q8_0 model, SHA-256
  `ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e`;
- 244-token production prompt and one generated token;
- Wave 118 kernel stack, including N16 prefetch, fused Q8 glue, compensated-MMA
  attention AV, and deferred FFN residual;
- Tesla T4 / `sm_75` hard selector;
- full CUDA parity before profiling.

## Measurements

1. Run one unprofiled production pass with on-stream CUDA-event telemetry and
   retain its eight-stage decomposition and correctness oracle.
2. Use Nsight Compute CLI application replay, `--cache-control none`, and a
   minimal metric list on the production GEMM and fused-attention entries.
3. Record duration, achieved occupancy, eligible/issued warp activity, tensor
   and shared-memory utilization, DRAM/L2 traffic, barrier stalls, long
   scoreboard stalls, and fixed-latency dependency stalls where supported.
4. Assemble the exact PTX with `ptxas -v` and disassemble the cubin so resource
   use and static instruction mix survive even when hosted workers deny access
   to performance counters.

Nsight Compute may replay a kernel and perturb its timing. Its durations are
diagnostic only and cannot replace the unprofiled production number. NVIDIA's
profiling guidance recommends limiting the selected kernels and metrics;
application replay avoids kernel-replay memory save/restore, while disabling
cache control preserves application-managed cache priming.

## Decision rule

The result must name one next mechanism and the evidence supporting it, or
state that the available counters cannot distinguish the candidates. A
counter-permission failure is a platform limitation, not a zero measurement.
In that case the next wave may improve `glbench`/the dedicated harness with
per-launch event distributions, but no performance optimization is licensed by
missing data alone.

Previously closed N32/M32, multi-stream prefill, native Q4, and M32 complete
prefetch directions remain closed. Wave 119 must not restack them.
