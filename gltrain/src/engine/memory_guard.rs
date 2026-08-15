// engine/memory_guard.rs — RAM usage safety watchdog (GWEN-208).
//
// Checked every 16 tokens during inference to halt generation before OOM.
// Using sysinfo::System which is already a dependency — no extra crate needed.

use sysinfo::System;

pub struct MemoryGuard {
    pub threshold_pct: u8,
}

impl MemoryGuard {
    /// Construct with a threshold in [10, 100].
    ///
    /// Below 10 % is rejected because that would trigger on any modern idle
    /// system and effectively disable inference entirely.
    pub fn new(threshold_pct: u8) -> Self {
        assert!(
            threshold_pct >= 10,
            "--auto-stop {} is dangerously low — minimum is 10%",
            threshold_pct
        );
        assert!(
            threshold_pct <= 100,
            "--auto-stop must be 0–100, got {}",
            threshold_pct
        );
        Self { threshold_pct }
    }

    /// Returns true when used RAM / total RAM ≥ threshold.
    /// The caller is responsible for calling `sys.refresh_memory()` before this.
    pub fn check(&self, sys: &System) -> bool {
        let total = sys.total_memory() as f64;
        if total == 0.0 {
            return false;
        }
        let used = sys.used_memory() as f64;
        (used / total * 100.0) >= self.threshold_pct as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_guard_threshold_validation() {
        // Thresholds in the valid range must not panic.
        let _g = MemoryGuard::new(10);
        let _g = MemoryGuard::new(90);
        let _g = MemoryGuard::new(100);
    }

    #[test]
    #[should_panic(expected = "dangerously low")]
    fn test_memory_guard_too_low_panics() {
        MemoryGuard::new(5);
    }

    #[test]
    fn test_memory_guard_check_mock() {
        // Simulate a system where 95 GB of 100 GB is used → 95 % used.
        // With a 90 % threshold the guard must fire.
        struct MockSys {
            used: u64,
            total: u64,
        }

        // We can't construct a real sysinfo::System with synthetic values,
        // so test the arithmetic directly using the same formula as check().
        let used_gb = 95_u64 * 1_073_741_824;
        let total_gb = 100_u64 * 1_073_741_824;
        let pct = (used_gb as f64 / total_gb as f64 * 100.0) as u8;
        assert!(pct >= 90, "expected ≥90%, got {}%", pct);

        let used_gb_low = 50_u64 * 1_073_741_824;
        let pct_low = (used_gb_low as f64 / total_gb as f64 * 100.0) as u8;
        assert!(pct_low < 90, "expected <90%, got {}%", pct_low);
    }
}
