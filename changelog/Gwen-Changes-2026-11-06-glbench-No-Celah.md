# Gwen Changes - 2026-11-06 - glbench No Celah

- Fixed `--profile stages` on glcuda being refused at every archive write, a
  dispatch fingerprint that missed the Wave 132 and 136 flags, and a decode
  ceiling built on the file size rather than the bytes the engine streams.
- Added report verbosity (`-q`, default, `-v`, `-vv`). Greedy runs a second
  instrumented pass and prints its measured cost: prefill -16.6%, decode -0.4%.
- Made `--kind` real: it was archived and read by nothing.
- Added a device read-bandwidth probe. The T4 measures 267 GB/s of a published
  320, which moves decode from 37% of an estimated ceiling to 44% of a measured
  one.
- Added `trend` and `list`, per-stage drift in `compare`, and comparison axes,
  so glproc against glcuda is no longer called a regression.
- Fixed CUDA stage rows being percentages of host DDR, and `total_ms` carrying
  no overhead because it was built from the phases it was compared against.
- Removed `runner/warmup.rs`.
- Replaced `wmic`, removed from Windows 11, with direct Win32 calls.
- Added time to first token and the model SHA-256 to archives.
- Made `ab` counterbalanced (A-B-B-A) with an order check and `--out`.
- Added NVML: clocks, temperature, power, GPU energy, VRAM and clock-limit
  reasons. Sampler cost in an A-B-B-A: -0.45% prefill, 0.0% decode.
- Found: the T4 sits on its 70 W cap (`sw_power_cap` in nearly every sample) and
  the SM clock falls under sustained load, 1,500 to about 1,100 MHz. Prefill
  tracks it at 10.2-10.7 tok/s per MHz. The 15,800 tok/s headline is a cool-card
  figure; sustained prefill-only load measures about 11,500.
- Added an order sketch to bit profiles: the Q6_K bug class scores 5.7e-2 where
  every distribution axis scores zero. The always-on tier goes from 27 to 54 ns
  per element.
- Added a per-token timeline, `--profile graph` with a production estimate in
  `-vv`, `glbench h2h` for llama-bench, and prefill per SM MHz.
- Measured: decode graph events cost 1.15x here, against 2.05x in Wave 133; the
  difference is not explained. Attention is 22.7% of decode, ffn_gate_up 20.9%,
  and 11.8% is outside the graph.
- Q4_K_M, repacked to Q8, beats Q8_0 in prefill per MHz by +3.9% and +3.7% across
  orders. Cause unknown.
- Results: `benchmarks/glbench-t4-wired-results.zip`,
  `benchmarks/glbench-t4-nvml-results.zip`,
  `benchmarks/glbench-t4-order-results.zip`.
