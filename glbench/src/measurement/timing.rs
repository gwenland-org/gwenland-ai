//! Timing primitives.
//!
//! A thin, dependency-free stopwatch over `std::time::Instant`, plus a
//! millisecond accessor. glbench mostly reads timing from the engine's
//! `InferOutput`, but the runner and any future in-process probe need a
//! consistent clock, and centralizing it keeps the unit (milliseconds, f64)
//! uniform across the crate.

use std::time::Instant;

/// A simple wall-clock stopwatch.
#[derive(Debug, Clone, Copy)]
pub struct Stopwatch {
    start: Instant,
}

impl Stopwatch {
    /// Start timing now.
    pub fn start() -> Stopwatch {
        Stopwatch { start: Instant::now() }
    }

    /// Elapsed milliseconds since [`Stopwatch::start`].
    pub fn elapsed_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1e3
    }
}
