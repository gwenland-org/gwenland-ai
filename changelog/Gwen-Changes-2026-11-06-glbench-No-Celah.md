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
- Results: `benchmarks/glbench-t4-wired-results.zip`,
  `benchmarks/glbench-t4-nvml-results.zip`.
