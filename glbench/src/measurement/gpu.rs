//! What the GPU was doing during the measured iterations.
//!
//! Throughput says how fast a run went; these say what state the card was in
//! while it did, which is the only way to tell a slower program from a slower
//! clock. A background thread samples NVML while the planner's measured loop
//! runs, so nothing is added between iterations. Energy needs no sampling: the
//! card keeps a counter, and two readings bracket the run exactly.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use glcuda::nvml::{reason_names, Nvml, NvmlSnapshot};

use crate::core::schema::{FromJson, ToJson};
use crate::export::json::Json;

/// Sampling period. NVML's own counters update at about this rate or slower.
const PERIOD: Duration = Duration::from_millis(50);

/// Minimum, mean and maximum of one sampled quantity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VLRange {
    pub min: f64,
    pub mean: f64,
    pub max: f64,
}

impl VLRange {
    fn of(values: impl Iterator<Item = f64>) -> Option<VLRange> {
        let values: Vec<f64> = values.collect();
        if values.is_empty() {
            return None;
        }
        Some(VLRange {
            min: values.iter().copied().fold(f64::INFINITY, f64::min),
            mean: values.iter().sum::<f64>() / values.len() as f64,
            max: values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        })
    }

    fn to_json(self) -> Json {
        Json::obj([
            ("min", Json::n(self.min)),
            ("mean", Json::n(self.mean)),
            ("max", Json::n(self.max)),
        ])
    }

    fn from_json(v: Option<&Json>) -> Option<VLRange> {
        let v = v?;
        let field = |k: &str| v.get(k).and_then(|n| n.as_f64());
        Some(VLRange { min: field("min")?, mean: field("mean")?, max: field("max")? })
    }
}

/// GPU state across the measured iterations.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct VLGpuRunStats {
    /// Samples taken. Zero with sampling off, when only energy is recorded.
    pub samples: usize,
    pub sm_clock_mhz: Option<VLRange>,
    pub mem_clock_mhz: Option<VLRange>,
    pub temperature_c: Option<VLRange>,
    pub power_w: Option<VLRange>,
    pub gpu_util_pct: Option<VLRange>,
    pub mem_util_pct: Option<VLRange>,
    /// Highest VRAM in use, by every process on the card.
    pub vram_used_peak_bytes: Option<u64>,
    /// The fastest and slowest P-state seen; `P0` is 0.
    pub pstate_range: Option<(u32, u32)>,
    /// Each clock-limit reason seen and how many samples it was set in.
    pub clock_reasons: Vec<(String, usize)>,
    /// Energy the card used, from its own counter.
    pub energy_joules: Option<f64>,
}

impl VLGpuRunStats {
    /// Reduce samples and the energy counter's two readings to run statistics.
    pub fn summarize(samples: &[NvmlSnapshot], energy_mj: Option<(u64, u64)>) -> VLGpuRunStats {
        let range = |f: fn(&NvmlSnapshot) -> Option<f64>| VLRange::of(samples.iter().filter_map(f));

        let mut clock_reasons: Vec<(String, usize)> = Vec::new();
        for bits in samples.iter().filter_map(|s| s.clock_reasons) {
            for name in reason_names(bits) {
                match clock_reasons.iter_mut().find(|(n, _)| n == name) {
                    Some((_, count)) => *count += 1,
                    None => clock_reasons.push((name.to_string(), 1)),
                }
            }
        }

        let pstates: Vec<u32> = samples.iter().filter_map(|s| s.pstate).collect();
        VLGpuRunStats {
            samples: samples.len(),
            sm_clock_mhz: range(|s| s.sm_clock_mhz.map(f64::from)),
            mem_clock_mhz: range(|s| s.mem_clock_mhz.map(f64::from)),
            temperature_c: range(|s| s.temperature_c.map(f64::from)),
            power_w: range(|s| s.power_mw.map(|mw| f64::from(mw) / 1e3)),
            gpu_util_pct: range(|s| s.gpu_util_pct.map(f64::from)),
            mem_util_pct: range(|s| s.mem_util_pct.map(f64::from)),
            vram_used_peak_bytes: samples.iter().filter_map(|s| s.vram_used_bytes).max(),
            pstate_range: pstates.iter().min().copied().zip(pstates.iter().max().copied()),
            clock_reasons,
            energy_joules: energy_mj.map(|(start, end)| end.saturating_sub(start) as f64 / 1e3),
        }
    }

    /// Clock-limit reasons that mean the card was held back while working.
    /// Idle and application clocks describe a choice, not a limit.
    pub fn limiting_reasons(&self) -> Vec<&(String, usize)> {
        self.clock_reasons
            .iter()
            .filter(|(name, _)| {
                !matches!(name.as_str(), "gpu_idle" | "applications_clocks" | "display_clock")
            })
            .collect()
    }
}

impl ToJson for VLGpuRunStats {
    fn to_json(&self) -> Json {
        let range = |r: Option<VLRange>| r.map_or(Json::Null, VLRange::to_json);
        Json::obj([
            ("samples", Json::n(self.samples as f64)),
            ("sm_clock_mhz", range(self.sm_clock_mhz)),
            ("mem_clock_mhz", range(self.mem_clock_mhz)),
            ("temperature_c", range(self.temperature_c)),
            ("power_w", range(self.power_w)),
            ("gpu_util_pct", range(self.gpu_util_pct)),
            ("mem_util_pct", range(self.mem_util_pct)),
            (
                "vram_used_peak_bytes",
                self.vram_used_peak_bytes.map_or(Json::Null, |b| Json::n(b as f64)),
            ),
            (
                "pstate_range",
                self.pstate_range.map_or(Json::Null, |(a, b)| {
                    Json::Arr(vec![Json::n(f64::from(a)), Json::n(f64::from(b))])
                }),
            ),
            (
                "clock_reasons",
                Json::Arr(
                    self.clock_reasons
                        .iter()
                        .map(|(name, count)| {
                            Json::obj([
                                ("reason", Json::s(name.clone())),
                                ("samples", Json::n(*count as f64)),
                            ])
                        })
                        .collect(),
                ),
            ),
            ("energy_joules", self.energy_joules.map_or(Json::Null, Json::n)),
        ])
    }
}

impl FromJson for VLGpuRunStats {
    fn from_json(v: &Json) -> Result<Self, String> {
        let range = |k: &str| VLRange::from_json(v.get(k).filter(|j| !matches!(j, Json::Null)));
        let pstate_range = v.get("pstate_range").and_then(|p| p.as_arr()).and_then(|p| {
            Some((p.first()?.as_f64()? as u32, p.get(1)?.as_f64()? as u32))
        });
        let clock_reasons = v
            .get("clock_reasons")
            .and_then(|r| r.as_arr())
            .map(|reasons| {
                reasons
                    .iter()
                    .filter_map(|r| {
                        let name = r.get("reason")?.as_str()?.to_string();
                        Some((name, r.get("samples")?.as_f64()? as usize))
                    })
                    .collect()
            })
            .unwrap_or_default();
        Ok(VLGpuRunStats {
            samples: v.get("samples").and_then(|n| n.as_f64()).unwrap_or(0.0) as usize,
            sm_clock_mhz: range("sm_clock_mhz"),
            mem_clock_mhz: range("mem_clock_mhz"),
            temperature_c: range("temperature_c"),
            power_w: range("power_w"),
            gpu_util_pct: range("gpu_util_pct"),
            mem_util_pct: range("mem_util_pct"),
            vram_used_peak_bytes: v
                .get("vram_used_peak_bytes")
                .and_then(|n| n.as_f64())
                .map(|n| n as u64),
            pstate_range,
            clock_reasons,
            energy_joules: v.get("energy_joules").and_then(|n| n.as_f64()),
        })
    }
}

/// A running observation of the card, from [`VLGpuMonitor::start`] until
/// [`VLGpuMonitor::finish`].
pub struct VLGpuMonitor {
    nvml: Arc<Nvml>,
    energy_start: Option<u64>,
    stop: Arc<AtomicBool>,
    sampler: Option<JoinHandle<Vec<NvmlSnapshot>>>,
}

impl VLGpuMonitor {
    /// Start observing. With `sample` off only the energy counter is read, at
    /// the two ends of the run, and nothing runs alongside it.
    pub fn start(nvml: Arc<Nvml>, sample: bool) -> VLGpuMonitor {
        let stop = Arc::new(AtomicBool::new(false));
        let sampler = sample.then(|| {
            let (nvml, stop) = (nvml.clone(), stop.clone());
            std::thread::spawn(move || {
                let mut samples = Vec::new();
                while !stop.load(Ordering::Relaxed) {
                    samples.push(nvml.snapshot());
                    std::thread::sleep(PERIOD);
                }
                samples
            })
        });
        VLGpuMonitor { energy_start: nvml.energy_mj(), nvml, stop, sampler }
    }

    /// Stop observing and reduce what was seen.
    pub fn finish(self) -> VLGpuRunStats {
        let energy_end = self.nvml.energy_mj();
        self.stop.store(true, Ordering::Relaxed);
        let samples = self.sampler.and_then(|s| s.join().ok()).unwrap_or_default();
        VLGpuRunStats::summarize(&samples, self.energy_start.zip(energy_end))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(sm: u32, reasons: u64) -> NvmlSnapshot {
        NvmlSnapshot {
            sm_clock_mhz: Some(sm),
            power_mw: Some(60_000),
            pstate: Some(0),
            clock_reasons: Some(reasons),
            vram_used_bytes: Some(u64::from(sm)),
            ..NvmlSnapshot::default()
        }
    }

    #[test]
    fn samples_reduce_to_ranges_and_absent_counters_stay_absent() {
        let stats = VLGpuRunStats::summarize(&[sample(1000, 0), sample(1500, 0)], Some((1000, 3500)));
        assert_eq!(stats.samples, 2);
        assert_eq!(stats.sm_clock_mhz, Some(VLRange { min: 1000.0, mean: 1250.0, max: 1500.0 }));
        assert_eq!(stats.power_w.map(|p| p.mean), Some(60.0));
        assert_eq!(stats.temperature_c, None, "an unsupported counter is absent, never zero");
        assert_eq!(stats.vram_used_peak_bytes, Some(1500));
        assert_eq!(stats.energy_joules, Some(2.5));
    }

    #[test]
    fn only_reasons_that_hold_the_card_back_count_as_limits() {
        let stats = VLGpuRunStats::summarize(
            &[sample(1000, 0x1), sample(1000, 0x1 | 0x4), sample(1000, 0x4)],
            None,
        );
        assert_eq!(
            stats.clock_reasons,
            [("gpu_idle".to_string(), 2), ("sw_power_cap".to_string(), 2)]
        );
        let limits = stats.limiting_reasons();
        assert_eq!(limits.len(), 1);
        assert_eq!(limits[0].0, "sw_power_cap");
    }

    #[test]
    fn stats_survive_the_archive() {
        let stats = VLGpuRunStats::summarize(&[sample(1000, 0x4)], Some((0, 1000)));
        let back = VLGpuRunStats::from_json(&stats.to_json()).unwrap();
        assert_eq!(back, stats);
    }

    #[test]
    fn energy_only_mode_has_no_samples() {
        let stats = VLGpuRunStats::summarize(&[], Some((0, 500)));
        assert_eq!(stats.samples, 0);
        assert_eq!(stats.sm_clock_mhz, None);
        assert_eq!(stats.energy_joules, Some(0.5));
    }
}
