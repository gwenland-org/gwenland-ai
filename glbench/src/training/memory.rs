//! [`VLTrainingMemory`] — what the run cost in memory.
//!
//! # Measured and derived are kept apart
//!
//! `peak_rss_bytes` is a platform read and is `None` wherever
//! [`crate::measurement::memory::peak_rss_bytes`] cannot answer. The parameter
//! and optimizer-state figures are *derived* from counts glbench already has,
//! and they are exact rather than estimated — `r*(d_in+d_out)` f32 values is
//! arithmetic, not a model.
//!
//! Mixing the two would be the mistake: a reader who cannot tell a measured RSS
//! from a computed footprint cannot tell whether a gap between them is
//! allocator overhead or an arithmetic error. So they sit in separate fields
//! with separate availability statuses.

use crate::core::schema::ToJson;
use crate::export::json::Json;

/// Bytes per f32.
const BYTES_PER_F32: u64 = 4;

/// Memory footprint of a training run.
#[derive(Debug, Clone, PartialEq)]
pub struct VLTrainingMemory {
    /// Resident set size before training started. `None` when the platform has
    /// no readable counter.
    pub rss_before_bytes: Option<u64>,
    /// Resident set size after the last step.
    pub rss_after_bytes: Option<u64>,
    /// Peak resident set size observed across the run. `None` on platforms
    /// without the counter — never an estimate.
    pub peak_rss_bytes: Option<u64>,
    /// `rss_after - rss_before`, when both are known. Derived.
    pub rss_growth_bytes: Option<i64>,

    /// Trainable parameter bytes at f32. Exact, derived from the adapter shape.
    pub parameter_bytes: u64,
    /// Optimizer state bytes at f32, from the real state tensors when the
    /// collector asked for them. `None` when it did not — the state exists, it
    /// simply was not read, which is `unavailable` rather than zero.
    pub optimizer_state_bytes: Option<u64>,
}

impl VLTrainingMemory {
    /// Build the record. `optimizer_state_elements` is `None` when the run did
    /// not request the tensor payload.
    pub fn new(
        rss_before_bytes: Option<u64>,
        rss_after_bytes: Option<u64>,
        peak_rss_bytes: Option<u64>,
        parameter_elements: usize,
        optimizer_state_elements: Option<usize>,
    ) -> VLTrainingMemory {
        VLTrainingMemory {
            rss_before_bytes,
            rss_after_bytes,
            peak_rss_bytes,
            // Signed: a run can end with a smaller RSS than it started with,
            // and reporting that as 0 would hide a real observation.
            rss_growth_bytes: match (rss_before_bytes, rss_after_bytes) {
                (Some(before), Some(after)) => Some(after as i64 - before as i64),
                _ => None,
            },
            parameter_bytes: parameter_elements as u64 * BYTES_PER_F32,
            optimizer_state_bytes: optimizer_state_elements
                .map(|e| e as u64 * BYTES_PER_F32),
        }
    }

    /// Dotted paths this value emits as `null`, with the honest reason.
    pub fn null_paths(&self) -> Vec<(&'static str, crate::core::availability::ENAvailability)> {
        use crate::core::availability::ENAvailability;
        let mut out = Vec::new();
        // The RSS counters travel together: if the platform has none, all four
        // are absent for the same reason.
        if self.rss_before_bytes.is_none() {
            out.push(("rss_before_bytes", ENAvailability::Unsupported));
        }
        if self.rss_after_bytes.is_none() {
            out.push(("rss_after_bytes", ENAvailability::Unsupported));
        }
        if self.peak_rss_bytes.is_none() {
            out.push(("peak_rss_bytes", ENAvailability::Unsupported));
        }
        if self.rss_growth_bytes.is_none() {
            out.push(("rss_growth_bytes", ENAvailability::Unsupported));
        }
        if self.optimizer_state_bytes.is_none() {
            out.push(("optimizer_state_bytes", ENAvailability::Unavailable));
        }
        out
    }
}

impl ToJson for VLTrainingMemory {
    fn to_json(&self) -> Json {
        let opt = |v: Option<u64>| v.map(|n| Json::n(n as f64)).unwrap_or(Json::Null);
        Json::obj([
            ("rss_before_bytes", opt(self.rss_before_bytes)),
            ("rss_after_bytes", opt(self.rss_after_bytes)),
            ("peak_rss_bytes", opt(self.peak_rss_bytes)),
            (
                "rss_growth_bytes",
                self.rss_growth_bytes.map(|n| Json::n(n as f64)).unwrap_or(Json::Null),
            ),
            ("parameter_bytes", Json::n(self.parameter_bytes as f64)),
            ("optimizer_state_bytes", opt(self.optimizer_state_bytes)),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::availability::ENAvailability;

    #[test]
    fn derived_footprints_are_exact_arithmetic_over_element_counts() {
        let m = VLTrainingMemory::new(None, None, None, 8192, Some(16384));
        assert_eq!(m.parameter_bytes, 8192 * 4);
        assert_eq!(m.optimizer_state_bytes, Some(16384 * 4));
    }

    #[test]
    fn rss_growth_is_derived_only_when_both_endpoints_are_known() {
        let m = VLTrainingMemory::new(Some(1000), Some(1500), Some(1600), 4, None);
        assert_eq!(m.rss_growth_bytes, Some(500));

        for (before, after) in [(Some(1000), None), (None, Some(1500)), (None, None)] {
            let m = VLTrainingMemory::new(before, after, None, 4, None);
            assert_eq!(m.rss_growth_bytes, None, "growth needs both endpoints");
        }
    }

    /// A run that frees more than it allocated is a real observation, not a
    /// zero.
    #[test]
    fn shrinking_memory_is_reported_as_negative_growth() {
        let m = VLTrainingMemory::new(Some(2000), Some(1200), Some(2100), 4, None);
        assert_eq!(m.rss_growth_bytes, Some(-800));
    }

    /// The distinction this module exists for: a counter the platform lacks is
    /// `unsupported`; state that exists but was not read is `unavailable`.
    #[test]
    fn absent_counters_and_unread_state_get_different_statuses() {
        let m = VLTrainingMemory::new(None, None, None, 16, None);
        let nulls = m.null_paths();

        for path in [
            "rss_before_bytes",
            "rss_after_bytes",
            "peak_rss_bytes",
            "rss_growth_bytes",
        ] {
            assert!(
                nulls.contains(&(path, ENAvailability::Unsupported)),
                "{path} must be unsupported"
            );
        }
        assert!(
            nulls.contains(&("optimizer_state_bytes", ENAvailability::Unavailable)),
            "unread state is unavailable, not unsupported"
        );
    }

    #[test]
    fn a_fully_measured_run_has_nothing_to_explain() {
        let m = VLTrainingMemory::new(Some(1000), Some(1500), Some(1600), 16, Some(32));
        assert!(m.null_paths().is_empty());
    }

    #[test]
    fn the_json_projection_nulls_exactly_what_null_paths_declares() {
        for m in [
            VLTrainingMemory::new(None, None, None, 16, None),
            VLTrainingMemory::new(Some(1), Some(2), Some(3), 16, Some(4)),
            VLTrainingMemory::new(Some(1), None, Some(3), 16, None),
        ] {
            let json = m.to_json();
            let obj = json.as_obj().unwrap();
            let actual: Vec<&str> = obj
                .iter()
                .filter(|(_, v)| matches!(v, Json::Null))
                .map(|(k, _)| k.as_str())
                .collect();
            let mut declared: Vec<&str> = m.null_paths().iter().map(|(p, _)| *p).collect();
            declared.sort_unstable();
            assert_eq!(actual, declared, "for {m:?}");
        }
    }

    /// `parameter_bytes` is derived from a count glbench already has, so it is
    /// never null and never needs a status.
    #[test]
    fn the_derived_parameter_footprint_is_always_present() {
        let m = VLTrainingMemory::new(None, None, None, 0, None);
        let json = m.to_json();
        assert!(!matches!(json.get("parameter_bytes"), Some(Json::Null)));
        assert!(!m.null_paths().iter().any(|(p, _)| *p == "parameter_bytes"));
    }
}
