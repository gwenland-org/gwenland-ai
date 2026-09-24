# Wave 119 - T4 kernel-profile result

## Outcome

Wave 119 reproduced the Wave 118 production dispatch and oracle on a Tesla T4,
assembled the production PTX with zero spills, and preserved a complete SASS
listing. Kaggle denied both requested Nsight Compute profiles with
`ERR_NVGPUCTRPERM`, so no hardware-counter value exists. This is a diagnostic
block, not evidence that any counter is zero.

The run exposed a second measurement defect. `GLCUDA_TELEMETRY=1` reported
102.310 ms for the 244-token pass while Wave 118's uninstrumented in-process
median was about 19.1 ms. Event timestamps are enqueued on stream, but
`prefill_batched` synchronously reads hundreds of event pairs before returning;
that host-side drain remains inside `GenTiming::prefill`. The stage timestamps
can rank work diagnostically, but the profiled wall time cannot be a production
throughput number.

## Verified facts

- GPU: Tesla T4, `sm_75`, driver 580.159.04.
- Model SHA-256:
  `ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e`.
- Oracle token: 3323.
- Gate/up dispatch: `bstage-n16-prefetch`.
- narrow/M32 dispatch: `bstage-n16-m32`.
- attention dispatch: `mma4-regq-avmma`.
- Main production module: zero stack and zero spills for every entry.
- Fused attention: 64 registers, one barrier, zero static shared memory
  (dynamic score storage is supplied at launch).
- Full CUDA parity passed before profiling.

## What `glbench` is missing

The next optimization cannot be selected honestly from current archived
telemetry. The missing observations are:

1. per-kernel or per-launch distributions across measured iterations rather
   than one last-iteration aggregate;
2. separate elementwise launches instead of one combined bucket;
3. QK, softmax, and AV attribution inside fused attention;
4. issue utilization, eligible warps, cache traffic, and stall reasons;
5. profiler overhead kept outside the production prefill timer;
6. resource assembly for opt-in auxiliary PTX modules as well as the main
   `glcuda_sm75.ptx` image.

NVIDIA documents that Nsight Compute may replay kernels, serialize launches,
and alter cache/clock behavior. Wave 119 therefore used application replay and
disabled cache flushing, but the hosted worker's counter policy prevented
collection. See the official [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/).

## Decision

Do not invent a Wave 120 kernel from the unavailable counters. Repair the
observation path first: retain a whole-prefill GPU event pair, drain detailed
events only after the production interval is closed, export distinct launch
families and distributions, and assemble every selected auxiliary PTX module.
Previously rejected N32/M32, M32 complete prefetch, multi-stream prefill, and
native-Q4 directions remain closed.

Evidence: `benchmarks/glcuda-t4-wave119-kernel-profile-results.zip`, SHA-256
`e60b0991979e164bd7c212e6b0e7b2cbc3328db9a161872264363fe3eb703409`.
