//! Per-phase timeline of a single request.
//!
//! Decomposes one iteration's wall-clock into load-independent phases (prefill,
//! decode) so a renderer can show where time went. The engine reports the two
//! phase durations directly; this is the small struct that carries them plus the
//! derived idle/overhead residual (total minus the accounted phases).

use crate::core::metrics::IterationMetrics;
use crate::core::schema::{field_f64, FromJson, ToJson};
use crate::export::json::Json;

/// A single request's time broken into phases, all in milliseconds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Timeline {
    /// Prefill (prompt processing) time.
    pub prefill_ms: f64,
    /// Decode (generation) time.
    pub decode_ms: f64,
    /// Wall-clock total.
    pub total_ms: f64,
}

impl Timeline {
    /// Build from a measured iteration.
    pub fn from_iteration(it: &IterationMetrics) -> Timeline {
        Timeline {
            prefill_ms: it.prefill_ms,
            decode_ms: it.decode_ms,
            total_ms: it.total_ms,
        }
    }

    /// Time unaccounted for by prefill + decode (sampling, detokenize, host
    /// overhead). Never negative.
    pub fn overhead_ms(&self) -> f64 {
        (self.total_ms - self.prefill_ms - self.decode_ms).max(0.0)
    }
}

/// Inter-token latency across the measured iterations, from the time each
/// token reached the caller.
#[derive(Debug, Clone, PartialEq)]
pub struct VLTokenTimeline {
    /// Median gap before each generated token, by position; index 0 is the gap
    /// between the first and second tokens. What this shows that a rate cannot
    /// is where in a generation the time goes, such as decode slowing as the
    /// context grows.
    pub median_gap_ms: Vec<f64>,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub max_ms: f64,
}

impl VLTokenTimeline {
    /// Reduce each iteration's gaps. `None` when no iteration produced two
    /// tokens.
    pub fn from_gaps(iterations: &[Vec<f64>]) -> Option<VLTokenTimeline> {
        let mut all: Vec<f64> = iterations.iter().flatten().copied().collect();
        if all.is_empty() {
            return None;
        }
        all.sort_by(f64::total_cmp);
        let longest = iterations.iter().map(Vec::len).max()?;
        let median_gap_ms = (0..longest)
            .map(|position| {
                let mut at: Vec<f64> =
                    iterations.iter().filter_map(|gaps| gaps.get(position).copied()).collect();
                at.sort_by(f64::total_cmp);
                percentile(&at, 0.5)
            })
            .collect();
        Some(VLTokenTimeline {
            median_gap_ms,
            p50_ms: percentile(&all, 0.5),
            p95_ms: percentile(&all, 0.95),
            p99_ms: percentile(&all, 0.99),
            max_ms: all[all.len() - 1],
        })
    }

    /// The median gap over `buckets` equal stretches of the generation.
    pub fn by_stretch(&self, buckets: usize) -> Vec<f64> {
        let n = self.median_gap_ms.len();
        let buckets = buckets.min(n).max(1);
        (0..buckets)
            .map(|b| {
                let mut part = self.median_gap_ms[b * n / buckets..(b + 1) * n / buckets].to_vec();
                part.sort_by(f64::total_cmp);
                percentile(&part, 0.5)
            })
            .collect()
    }
}

/// The value at fraction `q` of a sorted, non-empty slice.
fn percentile(sorted: &[f64], q: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * q).round() as usize]
}

impl ToJson for VLTokenTimeline {
    fn to_json(&self) -> Json {
        Json::obj([
            (
                "median_gap_ms",
                Json::Arr(self.median_gap_ms.iter().copied().map(Json::n).collect()),
            ),
            ("p50_ms", Json::n(self.p50_ms)),
            ("p95_ms", Json::n(self.p95_ms)),
            ("p99_ms", Json::n(self.p99_ms)),
            ("max_ms", Json::n(self.max_ms)),
        ])
    }
}

impl FromJson for VLTokenTimeline {
    fn from_json(v: &Json) -> Result<Self, String> {
        Ok(VLTokenTimeline {
            median_gap_ms: v
                .get("median_gap_ms")
                .and_then(|a| a.as_arr())
                .ok_or("missing 'median_gap_ms'")?
                .iter()
                .filter_map(|n| n.as_f64())
                .collect(),
            p50_ms: field_f64(v, "p50_ms")?,
            p95_ms: field_f64(v, "p95_ms")?,
            p99_ms: field_f64(v, "p99_ms")?,
            max_ms: field_f64(v, "max_ms")?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overhead_is_residual() {
        let it = IterationMetrics {
            prompt_tokens: 10,
            generated_tokens: 20,
            prefill_ms: 50.0,
            decode_ms: 400.0,
            total_ms: 470.0,
            ttft_ms: None,
        };
        let tl = Timeline::from_iteration(&it);
        assert!((tl.overhead_ms() - 20.0).abs() < 1e-9);
    }

    #[test]
    fn gaps_reduce_by_position_and_overall() {
        let t = VLTokenTimeline::from_gaps(&[vec![1.0, 4.0, 9.0], vec![3.0, 4.0]]).unwrap();
        assert_eq!(t.median_gap_ms, [3.0, 4.0, 9.0]);
        assert_eq!(t.max_ms, 9.0);
        assert_eq!(t.p50_ms, 4.0);
    }

    #[test]
    fn a_generation_of_one_token_has_no_timeline() {
        assert_eq!(VLTokenTimeline::from_gaps(&[vec![], vec![]]), None);
    }

    #[test]
    fn stretches_show_decode_slowing_down() {
        let t = VLTokenTimeline::from_gaps(&[(1..=8).map(f64::from).collect()]).unwrap();
        let stretches = t.by_stretch(4);
        assert_eq!(stretches.len(), 4);
        assert!(stretches[0] < stretches[3]);
        assert_eq!(t.by_stretch(100).len(), 8, "no more stretches than tokens");
    }

    #[test]
    fn the_timeline_survives_the_archive() {
        let t = VLTokenTimeline::from_gaps(&[vec![1.5, 2.5]]).unwrap();
        assert_eq!(VLTokenTimeline::from_json(&t.to_json()).unwrap(), t);
    }

    #[test]
    fn overhead_never_negative() {
        let it = IterationMetrics {
            prompt_tokens: 0,
            generated_tokens: 0,
            prefill_ms: 100.0,
            decode_ms: 100.0,
            total_ms: 150.0, // less than sum (clock skew)
            ttft_ms: None,
        };
        assert_eq!(Timeline::from_iteration(&it).overhead_ms(), 0.0);
    }
}
