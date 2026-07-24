//! Measured facts — and *only* facts.
//!
//! The cardinal rule (DESIGN.md, MEASUREMENT section): measurement stores raw
//! numbers, never conclusions. `memory_bandwidth = 240.0` belongs here;
//! `bottleneck = MemoryBound` does not — that is [`crate::analysis`]'s job,
//! derived from these facts. Keeping the two apart is what lets glbench claim
//! to measure *truth*: the facts are auditable and the interpretation is
//! separable from them.

use crate::core::schema::{field_f64, FromJson, ToJson};
use crate::export::json::Json;

/// A single timed iteration's raw counters, straight from the engine.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IterationMetrics {
    /// Prompt tokens processed during prefill.
    pub prompt_tokens: u64,
    /// Tokens generated during decode.
    pub generated_tokens: u64,
    /// Prefill wall-clock, milliseconds.
    pub prefill_ms: f64,
    /// Decode wall-clock, milliseconds.
    pub decode_ms: f64,
    /// Total request wall-clock, milliseconds.
    pub total_ms: f64,
}

impl IterationMetrics {
    /// Prefill throughput in tokens/second (0 if no prefill time recorded).
    pub fn prefill_tps(&self) -> f64 {
        rate(self.prompt_tokens as f64, self.prefill_ms)
    }

    /// Decode throughput in tokens/second (0 if no decode time recorded).
    pub fn decode_tps(&self) -> f64 {
        rate(self.generated_tokens as f64, self.decode_ms)
    }
}

/// Divide a count by a millisecond duration to get a per-second rate, guarding
/// division by zero.
fn rate(count: f64, ms: f64) -> f64 {
    if ms <= 0.0 {
        0.0
    } else {
        count / (ms / 1e3)
    }
}

impl ToJson for IterationMetrics {
    fn to_json(&self) -> Json {
        Json::obj([
            ("prompt_tokens", Json::n(self.prompt_tokens as f64)),
            ("generated_tokens", Json::n(self.generated_tokens as f64)),
            ("prefill_ms", Json::n(self.prefill_ms)),
            ("decode_ms", Json::n(self.decode_ms)),
            ("total_ms", Json::n(self.total_ms)),
        ])
    }
}

impl FromJson for IterationMetrics {
    fn from_json(v: &Json) -> Result<Self, String> {
        Ok(IterationMetrics {
            prompt_tokens: field_f64(v, "prompt_tokens")? as u64,
            generated_tokens: field_f64(v, "generated_tokens")? as u64,
            prefill_ms: field_f64(v, "prefill_ms")?,
            decode_ms: field_f64(v, "decode_ms")?,
            total_ms: field_f64(v, "total_ms")?,
        })
    }
}

/// The collected raw measurements for a workload: every timed iteration, plus
/// optional device-level facts sampled during the run. No derived verdicts.
#[derive(Debug, Clone, Default)]
pub struct MeasurementSet {
    /// One entry per measured iteration, in execution order.
    pub iterations: Vec<IterationMetrics>,
    /// Peak process/device memory observed during the run, bytes (if sampled).
    pub peak_memory_bytes: Option<u64>,
    /// Observed effective memory bandwidth, GB/s (if the engine or a probe
    /// reported it). A raw number — not a "bound" classification.
    pub observed_bandwidth_gbs: Option<f64>,
    /// Model file size on disk, bytes — the weight footprint that decode must
    /// stream. Used later as the numerator for a bandwidth-efficiency estimate.
    pub model_bytes: Option<u64>,
    /// Every cold-start iteration, timed but kept out of `iterations`: each
    /// pays page-in and cache-fill costs the warm numbers deliberately
    /// exclude, and mixing them in would corrupt both figures. Multiple
    /// iterations (not just the first) because a single cold sample cannot
    /// distinguish "this model always pays ~500ms to page in" from "the OS
    /// happened to schedule something else during this one run" — a
    /// median-and-range over several tells them apart. Empty when the run's
    /// `cold_iters` was 0.
    pub cold: Vec<IterationMetrics>,
    /// Package energy consumed across the measured iterations, Joules —
    /// measured via RAPL, never estimated from TDP. `None` when no energy
    /// counter was readable (non-Linux, missing permissions, no RAPL).
    pub energy_joules: Option<f64>,
    /// Average CPU utilization across the measured phase, percent, where
    /// 100% means one core continuously busy (see
    /// [`crate::measurement::cpu::utilization_pct`] for what values above
    /// 100 mean on a multi-core budget). `None` when the OS-level process
    /// CPU-time counters were not readable.
    pub cpu_utilization_pct: Option<f64>,
}

impl MeasurementSet {
    /// Number of recorded iterations.
    pub fn len(&self) -> usize {
        self.iterations.len()
    }

    /// True if no iterations were recorded.
    pub fn is_empty(&self) -> bool {
        self.iterations.is_empty()
    }

    /// Decode throughput samples across all iterations, tokens/second.
    pub fn decode_tps_samples(&self) -> Vec<f64> {
        self.iterations.iter().map(|m| m.decode_tps()).collect()
    }

    /// Prefill throughput samples across all iterations, tokens/second.
    pub fn prefill_tps_samples(&self) -> Vec<f64> {
        self.iterations.iter().map(|m| m.prefill_tps()).collect()
    }

    /// Measured Joules per generated token across the measured iterations.
    /// `None` unless energy was measured AND tokens were generated.
    pub fn joules_per_token(&self) -> Option<f64> {
        let tokens: u64 = self.iterations.iter().map(|i| i.generated_tokens).sum();
        let j = self.energy_joules?;
        (tokens > 0).then(|| j / tokens as f64)
    }

    /// Decode throughput samples across every cold-start iteration,
    /// tokens/second — the input to a median/range summary of "what a
    /// deployment's first few requests actually pay."
    pub fn cold_decode_tps_samples(&self) -> Vec<f64> {
        self.cold.iter().map(|m| m.decode_tps()).collect()
    }

    /// Prefill throughput samples across every cold-start iteration,
    /// tokens/second.
    pub fn cold_prefill_tps_samples(&self) -> Vec<f64> {
        self.cold.iter().map(|m| m.prefill_tps()).collect()
    }
}

impl ToJson for MeasurementSet {
    fn to_json(&self) -> Json {
        Json::obj([
            (
                "iterations",
                Json::Arr(self.iterations.iter().map(|i| i.to_json()).collect()),
            ),
            (
                "peak_memory_bytes",
                opt_num(self.peak_memory_bytes.map(|b| b as f64)),
            ),
            (
                "observed_bandwidth_gbs",
                opt_num(self.observed_bandwidth_gbs),
            ),
            ("model_bytes", opt_num(self.model_bytes.map(|b| b as f64))),
            (
                "cold",
                Json::Arr(self.cold.iter().map(|c| c.to_json()).collect()),
            ),
            ("energy_joules", opt_num(self.energy_joules)),
            ("cpu_utilization_pct", opt_num(self.cpu_utilization_pct)),
        ])
    }
}

impl FromJson for MeasurementSet {
    fn from_json(v: &Json) -> Result<Self, String> {
        let iters = v
            .get("iterations")
            .and_then(|a| a.as_arr())
            .ok_or("missing 'iterations' array")?
            .iter()
            .map(IterationMetrics::from_json)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(MeasurementSet {
            iterations: iters,
            peak_memory_bytes: v.get("peak_memory_bytes").and_then(|n| n.as_f64()).map(|n| n as u64),
            observed_bandwidth_gbs: v.get("observed_bandwidth_gbs").and_then(|n| n.as_f64()),
            model_bytes: v.get("model_bytes").and_then(|n| n.as_f64()).map(|n| n as u64),
            // Both shapes accepted: pre-multi-cold-start archives wrote a
            // single object (or null), current ones write an array. Either
            // way, absent reads as "no cold-start phase was run" rather than
            // a parse failure — reading an old archive must not become an
            // error just because this crate learned to record more of them.
            cold: match v.get("cold") {
                Some(c) if c.as_arr().is_some() => c
                    .as_arr()
                    .unwrap()
                    .iter()
                    .filter_map(|i| IterationMetrics::from_json(i).ok())
                    .collect(),
                Some(c) => IterationMetrics::from_json(c).ok().into_iter().collect(),
                None => Vec::new(),
            },
            energy_joules: v.get("energy_joules").and_then(|n| n.as_f64()),
            cpu_utilization_pct: v.get("cpu_utilization_pct").and_then(|n| n.as_f64()),
        })
    }
}

/// Encode an optional number: the value, or JSON null when absent.
fn opt_num(v: Option<f64>) -> Json {
    match v {
        Some(n) => Json::n(n),
        None => Json::Null,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tps_guards_zero_time() {
        let m = IterationMetrics {
            prompt_tokens: 100,
            generated_tokens: 50,
            prefill_ms: 0.0,
            decode_ms: 0.0,
            total_ms: 0.0,
        };
        assert_eq!(m.prefill_tps(), 0.0);
        assert_eq!(m.decode_tps(), 0.0);
    }

    #[test]
    fn tps_math() {
        let m = IterationMetrics {
            prompt_tokens: 500,
            generated_tokens: 128,
            prefill_ms: 500.0, // 0.5 s -> 1000 tps
            decode_ms: 4000.0, // 4.0 s -> 32 tps
            total_ms: 4500.0,
        };
        assert!((m.prefill_tps() - 1000.0).abs() < 1e-6);
        assert!((m.decode_tps() - 32.0).abs() < 1e-6);
    }

    #[test]
    fn measurement_round_trips() {
        let set = MeasurementSet {
            iterations: vec![IterationMetrics {
                prompt_tokens: 10,
                generated_tokens: 20,
                prefill_ms: 5.0,
                decode_ms: 100.0,
                total_ms: 105.0,
            }],
            peak_memory_bytes: Some(1 << 30),
            observed_bandwidth_gbs: Some(240.5),
            model_bytes: Some(4_400_000_000),
            cold: vec![IterationMetrics {
                prompt_tokens: 10,
                generated_tokens: 20,
                prefill_ms: 50.0, // cold prefill pays page-in: 10x the warm one
                decode_ms: 150.0,
                total_ms: 200.0,
            }],
            energy_joules: Some(41.5),
            cpu_utilization_pct: Some(65.5),
        };
        let back = MeasurementSet::from_json(&set.to_json()).unwrap();
        assert_eq!(back.iterations, set.iterations);
        assert_eq!(back.peak_memory_bytes, set.peak_memory_bytes);
        assert_eq!(back.observed_bandwidth_gbs, set.observed_bandwidth_gbs);
        assert_eq!(back.model_bytes, set.model_bytes);
        assert_eq!(back.cold, set.cold);
        assert_eq!(back.energy_joules, set.energy_joules);
    }

    #[test]
    fn a_pre_multi_cold_start_archive_with_a_bare_cold_object_still_parses() {
        // Archives written before cold became a Vec stored a single object
        // (or null). Reading one must not become an error.
        let json = Json::obj([
            ("iterations", Json::Arr(vec![])),
            ("peak_memory_bytes", Json::Null),
            ("observed_bandwidth_gbs", Json::Null),
            ("model_bytes", Json::Null),
            (
                "cold",
                IterationMetrics {
                    prompt_tokens: 1,
                    generated_tokens: 2,
                    prefill_ms: 3.0,
                    decode_ms: 4.0,
                    total_ms: 5.0,
                }
                .to_json(),
            ),
            ("energy_joules", Json::Null),
        ]);
        let back = MeasurementSet::from_json(&json).unwrap();
        assert_eq!(back.cold.len(), 1);
        assert_eq!(back.cold[0].generated_tokens, 2);
    }

    #[test]
    fn an_archive_with_no_cold_field_at_all_parses_to_an_empty_vec() {
        let json = Json::obj([
            ("iterations", Json::Arr(vec![])),
            ("peak_memory_bytes", Json::Null),
            ("observed_bandwidth_gbs", Json::Null),
            ("model_bytes", Json::Null),
            ("energy_joules", Json::Null),
        ]);
        let back = MeasurementSet::from_json(&json).unwrap();
        assert!(back.cold.is_empty());
    }

    #[test]
    fn joules_per_token_needs_both_energy_and_tokens() {
        let mut set = MeasurementSet::default();
        assert!(set.joules_per_token().is_none(), "no energy, no figure");
        set.energy_joules = Some(50.0);
        assert!(set.joules_per_token().is_none(), "energy but zero tokens");
        set.iterations.push(IterationMetrics {
            prompt_tokens: 5,
            generated_tokens: 25,
            prefill_ms: 10.0,
            decode_ms: 100.0,
            total_ms: 110.0,
        });
        assert!((set.joules_per_token().unwrap() - 2.0).abs() < 1e-9);
    }
}
