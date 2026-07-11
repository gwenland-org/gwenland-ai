//! Ceiling analysis: compare observed performance against the hardware's
//! theoretical capability.
//!
//! For memory-bound decode, the ceiling is bandwidth. A model of size `W` bytes
//! that streams its full weight set once per generated token cannot exceed
//! `peak_bandwidth / W` tokens/second. Comparing observed decode tok/s against
//! that number yields an efficiency fraction — the single most useful "how much
//! of the machine are we using" figure. If no peak bandwidth is known (CPU run,
//! or an unrecognized GPU), the ceiling is simply unavailable and glbench says
//! so rather than inventing one.

use crate::comparison::statistics::Stats;
use crate::core::session::BenchmarkSession;

/// The result of ceiling analysis.
#[derive(Debug, Clone, Default)]
pub struct Ceiling {
    /// The theoretical decode ceiling in tokens/second, if it could be
    /// established.
    pub decode_tps_ceiling: Option<f64>,
    /// Observed / ceiling, 0.0..=1.0, if a ceiling exists.
    pub efficiency: Option<f64>,
    /// The bandwidth used as the ceiling basis, GB/s.
    pub basis_bandwidth_gbs: Option<f64>,
    /// Observations to surface to the user.
    pub notes: Vec<String>,
}

/// Establish the decode bandwidth ceiling for `session` and compare the
/// observed decode throughput (`decode_tps`) against it.
pub fn analyze(session: &BenchmarkSession, decode_tps: &Stats) -> Ceiling {
    let mut c = Ceiling::default();

    // Weight footprint decode must stream: prefer the measured model bytes,
    // else the model file size on disk.
    let model_bytes = session
        .measurements
        .model_bytes
        .or(session.environment.hardware.storage.model_file_bytes);

    // Ceiling basis: the device's peak bandwidth if a GPU reported one,
    // otherwise fall back to any observed bandwidth the run captured.
    let peak_bw = session
        .environment
        .hardware
        .gpu
        .peak_bandwidth_gbs
        .or(session.measurements.observed_bandwidth_gbs);

    let (Some(bytes), Some(bw_gbs)) = (model_bytes, peak_bw) else {
        c.notes.push(
            "No hardware bandwidth ceiling available (CPU run or unknown device); \
             efficiency vs peak cannot be computed."
                .to_string(),
        );
        return c;
    };

    // peak bytes/s = bw_gbs * 1e9; tokens/s ceiling = bytes_per_s / model_bytes.
    let ceiling = (bw_gbs * 1e9) / bytes as f64;
    c.decode_tps_ceiling = Some(ceiling);
    c.basis_bandwidth_gbs = Some(bw_gbs);

    if decode_tps.mean > 0.0 && ceiling > 0.0 {
        let eff = (decode_tps.mean / ceiling).clamp(0.0, 1.0);
        c.efficiency = Some(eff);
        c.notes.push(format!(
            "Decode {:.1} tok/s vs bandwidth ceiling {:.1} tok/s ({:.0} GB/s over {:.2} GB weights) = {:.0}% of peak.",
            decode_tps.mean,
            ceiling,
            bw_gbs,
            bytes as f64 / 1e9,
            eff * 100.0,
        ));
    }
    c
}
