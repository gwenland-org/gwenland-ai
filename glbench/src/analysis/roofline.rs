//! Roofline estimate: is a workload on the memory-bound or compute-bound side
//! of the roofline, given its arithmetic intensity?
//!
//! For a token-decode workload streaming `W` bytes of weights and doing ~`2·W/bpw`
//! FLOPs of matmul against them, arithmetic intensity is low (well under 1
//! FLOP/byte for the low-batch decode case), which places decode firmly on the
//! bandwidth-bound side — the roofline confirmation of the ceiling analysis.
//! This module gives the arithmetic-intensity number; it does not re-derive the
//! bottleneck (that is [`super::bottleneck`]'s job).

/// The ridge point of a roofline and where a workload sits relative to it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Roofline {
    /// Arithmetic intensity of the workload, FLOP per byte.
    pub intensity_flop_per_byte: f64,
    /// The machine balance (ridge point): peak_compute / peak_bandwidth,
    /// FLOP per byte. Below this the workload is bandwidth-bound.
    pub ridge_flop_per_byte: f64,
}

impl Roofline {
    /// True if the workload is on the memory-bound side of the ridge.
    pub fn is_memory_bound(&self) -> bool {
        self.intensity_flop_per_byte < self.ridge_flop_per_byte
    }

    /// Build from peak compute (FLOP/s) and peak bandwidth (bytes/s) plus the
    /// workload's FLOP and byte counts.
    pub fn new(
        workload_flops: f64,
        workload_bytes: f64,
        peak_flops: f64,
        peak_bytes_per_s: f64,
    ) -> Option<Roofline> {
        if workload_bytes <= 0.0 || peak_bytes_per_s <= 0.0 {
            return None;
        }
        Some(Roofline {
            intensity_flop_per_byte: workload_flops / workload_bytes,
            ridge_flop_per_byte: peak_flops / peak_bytes_per_s,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn low_intensity_is_memory_bound() {
        // decode: ~0.25 FLOP/byte, ridge at ~140 (T4 65 TFLOP / 320 GB/s).
        let r = Roofline::new(1.0, 4.0, 65e12, 320e9).unwrap();
        assert!(r.is_memory_bound());
    }
}
