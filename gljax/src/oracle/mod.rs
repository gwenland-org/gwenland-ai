//! Correctness oracles (ARTX12 Part B §2 — "Oracle Architecture").
//!
//! ARTX12 separates *four* oracle tiers because a single oracle cannot tell
//! "wrong algorithm" apart from "different precision" apart from "different
//! model":
//!
//! ```text
//! T0  Pure-Rust f64 reference          "Is the algorithm right?"
//! T1  StableHLO reference interpreter  "Did we emit the MLIR we think we did?"
//! T2  FP64 CPU plugin (ARTX01 §3.4)    "Does the real compiler+runtime agree?"
//! T3  Cross-engine differential        "Do we diverge from another engine?"
//! ```
//!
//! [`reference`] is T0. T2 already exists in spirit — `PrecisionPolicy::f64_oracle`
//! plus a real PJRT plugin is exactly T2 — but this environment has no plugin
//! to run it against (`gljax/README.md`). T1 and T3 are not built here: T1
//! needs the `stablehlo-translate` binary (see `gljax/tests/t1_interpreter.rs`,
//! which is a real, working harness that SKIPs cleanly without it — the same
//! pattern every PJRT-gated test in this crate already uses); T3 needs a
//! second engine (glproc) wired to the same checkpoint, which is out of
//! scope for this wave.

pub mod reference;
