//! Memory-usage measurement helpers.
//!
//! glbench does not link a GPU SDK, so device-memory peak comes from the engine
//! (if it reports one) rather than a driver query here. This module holds the
//! host-side facts std can observe — currently the model file size, which is the
//! resident weight footprint a fully-loaded engine holds. Kept separate from
//! [`crate::environment::storage`] because that is the *snapshot* field; this is
//! the *measurement* helper the runner uses.

/// Convert a byte count to gibibytes for display.
pub fn bytes_to_gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_gib() {
        assert!((bytes_to_gib(1u64 << 30) - 1.0).abs() < 1e-9);
    }
}
