//! Runtime SIMD backend selection.
//!
//! Detection runs exactly once (cached in a `OnceLock`) — never probe CPU
//! features in the hot path.

use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdStrategy {
    Scalar,
    Avx2,
    Avx512,
}

static DETECTED: OnceLock<SimdStrategy> = OnceLock::new();

impl SimdStrategy {
    /// The backend for this machine. Detects on first call, then returns the
    /// cached value (a single atomic load) — safe to call from dispatchers.
    pub fn detect() -> Self {
        *DETECTED.get_or_init(Self::probe)
    }

    /// Probe CPU features and core count. Called once per process.
    fn probe() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            let avx512 = std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512bw");
            // f16c is required because the wide kernels convert block scales
            // with `vcvtph2ps`. Every AVX2+FMA part ships it (F16C predates
            // AVX2), but gate on it anyway so the unsafe contract is airtight.
            let avx2 = std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
                && std::arch::is_x86_feature_detected!("f16c");

            // AVX-512 heuristic: only use it on parts with more than 8 logical
            // cores (desktop/server). On mobile TDP (e.g. i3-1115G4, 15 W),
            // AVX-512 triggers a frequency throttle that makes 4-thread AVX2
            // at ~3.5 GHz faster than AVX-512 at ~2.5 GHz.
            let is_likely_laptop = num_cpus::get() <= 8;

            if avx512 && !is_likely_laptop {
                return SimdStrategy::Avx512;
            }
            if avx2 {
                return SimdStrategy::Avx2;
            }
            if avx512 {
                // AVX-512 present but AVX2+FMA somehow not detected — take it.
                return SimdStrategy::Avx512;
            }
        }
        SimdStrategy::Scalar
    }
}
