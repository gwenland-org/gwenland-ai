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

/// Where the bandwidth ceiling number came from — the distinction that
/// decides how much weight a reader should put on `efficiency`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CeilingBasis {
    /// No ceiling could be established at all — not a GPU run with a known
    /// device, no CPU bandwidth probe result, nothing observed.
    #[default]
    Undetermined,
    /// The CPU's own sustained sequential-read bandwidth, measured on this
    /// machine at session start (`environment::bandwidth`). The honest
    /// number: it already reflects this machine's actual RAM, channel
    /// count, and thermal state, not a spec sheet.
    Measured,
    /// A GPU's published peak bandwidth from `engine::capability`'s static
    /// table — a vendor specification, not something this session measured.
    /// Real GPUs commonly run 60-85% of their advertised peak even on a
    /// well-optimized kernel, so an efficiency computed against this basis
    /// is being compared to a number no run of this device will ever fully
    /// reach.
    EstimatedFromTable,
}

impl CeilingBasis {
    /// Stable identifier, used in JSON and as a table/log label.
    pub fn as_str(self) -> &'static str {
        match self {
            CeilingBasis::Undetermined => "undetermined",
            CeilingBasis::Measured => "measured",
            CeilingBasis::EstimatedFromTable => "estimated_from_table",
        }
    }
}

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
    /// Where `basis_bandwidth_gbs` came from — measured on this machine, a
    /// vendor's published spec, or nothing at all.
    pub basis: CeilingBasis,
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

    // Ceiling basis, in order of preference — each candidate paired with
    // what kind of number it is, so the winner's provenance travels with it:
    //   1. anything THIS RUN observed directly (measured, if ever populated);
    //   2. the CPU's MEASURED sequential read bandwidth (environment probe);
    //   3. the GPU's published peak — a vendor spec, not a measurement.
    //
    // CPU beats the GPU table in priority for exactly the reason a table entry
    // is weaker evidence: DDR4-2667 single- vs dual-channel differ 2x and no
    // CPUID bit distinguishes them, so a *measured* number is trusted first
    // regardless of device kind. This is also why CPU runs used to report
    // `bottleneck: undetermined` forever — the analysis used to look only at
    // the GPU table, so a CPU run never had a ceiling and the whole roofline
    // stayed dark.
    let winner: Option<(f64, CeilingBasis)> = session
        .measurements
        .observed_bandwidth_gbs
        .map(|bw| (bw, CeilingBasis::Measured))
        .or_else(|| {
            session
                .environment
                .hardware
                .cpu
                .read_bandwidth_gbs
                .map(|bw| (bw, CeilingBasis::Measured))
        })
        .or_else(|| {
            session
                .environment
                .hardware
                .gpu
                .peak_bandwidth_gbs
                .map(|bw| (bw, CeilingBasis::EstimatedFromTable))
        });
    let peak_bw = winner.map(|(bw, _)| bw);
    let basis = winner.map(|(_, b)| b);

    let (Some(bytes), Some(bw_gbs)) = (model_bytes, peak_bw) else {
        c.notes.push(
            "No bandwidth ceiling available; efficiency vs peak cannot be computed.".to_string(),
        );
        c.basis = CeilingBasis::Undetermined;
        return c;
    };
    c.basis = basis.unwrap_or(CeilingBasis::Undetermined);

    // peak bytes/s = bw_gbs * 1e9; tokens/s ceiling = bytes_per_s / model_bytes.
    let ceiling = (bw_gbs * 1e9) / bytes as f64;
    c.decode_tps_ceiling = Some(ceiling);
    c.basis_bandwidth_gbs = Some(bw_gbs);

    if decode_tps.mean > 0.0 && ceiling > 0.0 {
        let eff = (decode_tps.mean / ceiling).clamp(0.0, 1.0);
        c.efficiency = Some(eff);
        let basis_note = match c.basis {
            CeilingBasis::Measured => {
                format!("Decode {:.1} tok/s vs bandwidth ceiling {:.1} tok/s ({:.0} GB/s, measured on this machine, over {:.2} GB weights) = {:.0}% of peak.",
                    decode_tps.mean, ceiling, bw_gbs, bytes as f64 / 1e9, eff * 100.0)
            }
            CeilingBasis::EstimatedFromTable => {
                format!("Decode {:.1} tok/s vs bandwidth ceiling {:.1} tok/s ({:.0} GB/s, the device's PUBLISHED spec, not measured, over {:.2} GB weights) = {:.0}% of peak. \
                    Real devices commonly run 60-85% of their advertised peak even when fully optimized, so this fraction is an underestimate of how well-utilized the device actually is.",
                    decode_tps.mean, ceiling, bw_gbs, bytes as f64 / 1e9, eff * 100.0)
            }
            CeilingBasis::Undetermined => unreachable!("efficiency requires a basis"),
        };
        c.notes.push(basis_note);
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::metrics::{IterationMetrics, MeasurementSet};
    use crate::core::result::SessionMetadata;
    use crate::core::workload::WorkloadSpec;
    use crate::engine::metadata::EngineMetadata;
    use crate::environment::gpu::GpuInfo;
    use crate::environment::hardware::EnvironmentSnapshot;

    fn sample(measure_decode_tps: f64) -> BenchmarkSession {
        let mut m = MeasurementSet::default();
        m.iterations.push(IterationMetrics {
            prompt_tokens: 10,
            generated_tokens: 100,
            prefill_ms: 10.0,
            decode_ms: (100.0 / measure_decode_tps) * 1000.0,
            total_ms: 0.0,
        });
        m.model_bytes = Some(1_000_000_000); // 1 GB
        BenchmarkSession::new(
            SessionMetadata::new("test-run"),
            EnvironmentSnapshot::probe(""),
            EngineMetadata {
                name: "glproc".into(),
                backend: "cpu".into(),
                available: true,
                model_arch: None,
                quantization: None,
                thinking_capable: None,
            },
            WorkloadSpec::default(),
            m,
        )
    }

    #[test]
    fn no_bandwidth_anywhere_is_undetermined() {
        let mut sess = sample(10.0);
        sess.environment.hardware.cpu.read_bandwidth_gbs = None;
        sess.environment.hardware.gpu = GpuInfo::default();
        let dec = Stats::from_samples(&sess.measurements.decode_tps_samples());
        let c = analyze(&sess, &dec);
        assert_eq!(c.basis, CeilingBasis::Undetermined);
        assert_eq!(c.efficiency, None);
    }

    #[test]
    fn cpu_measured_bandwidth_wins_over_no_gpu() {
        let mut sess = sample(10.0);
        sess.environment.hardware.cpu.read_bandwidth_gbs = Some(20.0);
        let dec = Stats::from_samples(&sess.measurements.decode_tps_samples());
        let c = analyze(&sess, &dec);
        assert_eq!(c.basis, CeilingBasis::Measured);
        assert!(c.efficiency.is_some());
    }

    #[test]
    fn cpu_measured_bandwidth_wins_over_gpu_table_even_when_both_present() {
        // A measured number beats a vendor table regardless of device kind
        // (see the module's own reasoning: DDR4 channel count is invisible
        // to CPUID, so a measured figure is trusted first).
        let mut sess = sample(10.0);
        sess.environment.hardware.cpu.read_bandwidth_gbs = Some(20.0);
        sess.environment.hardware.gpu.peak_bandwidth_gbs = Some(900.0);
        let dec = Stats::from_samples(&sess.measurements.decode_tps_samples());
        let c = analyze(&sess, &dec);
        assert_eq!(c.basis, CeilingBasis::Measured);
        assert_eq!(c.basis_bandwidth_gbs, Some(20.0));
    }

    #[test]
    fn gpu_table_is_the_fallback_when_nothing_was_measured() {
        let mut sess = sample(10.0);
        sess.environment.hardware.cpu.read_bandwidth_gbs = None;
        sess.environment.hardware.gpu.peak_bandwidth_gbs = Some(900.0);
        let dec = Stats::from_samples(&sess.measurements.decode_tps_samples());
        let c = analyze(&sess, &dec);
        assert_eq!(c.basis, CeilingBasis::EstimatedFromTable);
        assert!(c.efficiency.is_some());
    }

    #[test]
    fn estimated_basis_note_warns_about_the_underestimate() {
        let mut sess = sample(10.0);
        sess.environment.hardware.cpu.read_bandwidth_gbs = None;
        sess.environment.hardware.gpu.peak_bandwidth_gbs = Some(900.0);
        let dec = Stats::from_samples(&sess.measurements.decode_tps_samples());
        let c = analyze(&sess, &dec);
        assert!(c.notes.iter().any(|n| n.contains("PUBLISHED spec, not measured")));
        assert!(c.notes.iter().any(|n| n.contains("underestimate")));
    }

    #[test]
    fn measured_basis_note_says_measured_not_estimated() {
        let mut sess = sample(10.0);
        sess.environment.hardware.cpu.read_bandwidth_gbs = Some(20.0);
        let dec = Stats::from_samples(&sess.measurements.decode_tps_samples());
        let c = analyze(&sess, &dec);
        assert!(c.notes.iter().any(|n| n.contains("measured on this machine")));
    }

    #[test]
    fn ceiling_basis_as_str_is_stable() {
        assert_eq!(CeilingBasis::Undetermined.as_str(), "undetermined");
        assert_eq!(CeilingBasis::Measured.as_str(), "measured");
        assert_eq!(CeilingBasis::EstimatedFromTable.as_str(), "estimated_from_table");
    }
}
