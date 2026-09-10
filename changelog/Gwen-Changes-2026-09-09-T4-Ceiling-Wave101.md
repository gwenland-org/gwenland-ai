# Gwen Changes - 2026-09-09 - T4 Ceiling Wave 101

- Made the Kaggle gate discover Cargo from `PATH` and common locations, with an
  isolated minimal Rust bootstrap when the image provides no toolchain.
- Confirmed the run used Tesla T4/sm_75 and passed all 65 host tests.
- Retained the N16 register-prefetch candidate after two bit-exact direct runs
  at 1.1989x and 1.2730x versus the current N16 kernel.
- Preserved the candidate behind its opt-in pending a full production prefill
  benchmark; no production tok/s claim changed in this wave.
