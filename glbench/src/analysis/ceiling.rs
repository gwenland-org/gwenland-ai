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

/// Backend identifiers that mean "the weights were streamed from device
/// memory, not host RAM".
///
/// Matched against `EngineMetadata::backend`, which the adapter fills from the
/// engine's own `EngineSpec` — so it states where the model actually ran
/// rather than what hardware happens to be present.
const GPU_BACKENDS: &[&str] = &["cuda", "vulkan", "metal", "rocm"];

/// Did this session run on a GPU?
fn ran_on_gpu(session: &BenchmarkSession) -> bool {
    let backend = session.engine.backend.to_ascii_lowercase();
    GPU_BACKENDS.iter().any(|b| backend == *b)
}

/// The memory bandwidth that bounds this run, paired with what kind of number
/// it is so the winner's provenance travels with it.
///
/// The ceiling must come from the memory the weights were actually streamed
/// from. Which device that was is not a preference — it is a fact the engine
/// reports, and reading it wrong produces a confidently wrong answer.
///
/// # Why this is one function and not two
///
/// [`analyze`] and the per-bucket roofline in [`super::summary`] used to
/// select independently. When the rule below was corrected, the roofline kept
/// the old one and went on classifying every stage of a CUDA run against host
/// DDR bandwidth — the same defect, surviving its own fix. One selection, two
/// callers.
///
/// # The defect this encodes against
///
/// ⛔ The rule used to be "any measured number beats any table entry,
/// regardless of device kind". On a CUDA run that picked the HOST's DDR
/// bandwidth over the GPU's, because the host figure was measured and the
/// device figure came from a table. Measured on a Tesla T4 (2026-08-20):
/// decode 210.9 tok/s against a ceiling of 54.1 tok/s built from 27 GB/s of
/// host RAM — 390% of a ceiling 12x too low, for a card whose 320 GB/s glbench
/// had already probed and printed one line earlier.
///
/// The old rationale was sound for its actual question (DDR4 single- vs
/// dual-channel differ 2x and no CPUID bit distinguishes them, so a measured
/// host figure beats a host *table* entry) and was wrongly generalised across
/// device kinds. Device first, then provenance within that device.
pub fn bandwidth_for_run(session: &BenchmarkSession) -> Option<(f64, CeilingBasis)> {
    let on_gpu = ran_on_gpu(session);
    session
        .measurements
        .observed_bandwidth_gbs
        .map(|bw| (bw, CeilingBasis::Measured))
        .or_else(|| {
            if on_gpu {
                // A vendor spec for the right device beats a measurement of the
                // wrong one. The note the caller emits says it is a spec.
                session
                    .environment
                    .hardware
                    .gpu
                    .peak_bandwidth_gbs
                    .map(|bw| (bw, CeilingBasis::EstimatedFromTable))
            } else {
                session
                    .environment
                    .hardware
                    .cpu
                    .read_bandwidth_gbs
                    .map(|bw| (bw, CeilingBasis::Measured))
            }
        })
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

    let winner = bandwidth_for_run(session);
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
        let raw = decode_tps.mean / ceiling;

        // ⛔ Nothing streams faster than the memory it streams from. A ratio
        // above 1 does not mean a spectacular result; it means the ceiling is
        // wrong, and every conclusion drawn from it is unfounded.
        //
        // This used to be `.clamp(0.0, 1.0)`, which turned an impossible 390%
        // into a perfect-looking "100% of peak" and reported `memory_bound`
        // with a recommendation to quantise harder. The clamp is exactly what
        // stopped `bench-skills/measurement-discipline.md`'s roofline sanity
        // check from ever firing. Report no efficiency instead: an unusable
        // number is not a number.
        if raw > 1.0 {
            c.efficiency = None;
            c.notes.push(format!(
                "IMPLAUSIBLE: decode {:.1} tok/s is {:.0}% of the computed ceiling \
                 {:.1} tok/s ({:.0} GB/s over {:.2} GB weights). Nothing streams faster \
                 than its own memory, so the ceiling is wrong — most likely the \
                 bandwidth figure describes a different device than the one that ran \
                 the model. No efficiency, bottleneck or roofline verdict is reported \
                 from it.",
                decode_tps.mean, raw * 100.0, ceiling, bw_gbs, bytes as f64 / 1e9
            ));
            return c;
        }

        let eff = raw;
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

    /// The same fixture, but reporting that it ran on a CUDA device. The only
    /// difference that matters is `engine.backend` — that field is what tells
    /// the analysis which memory the weights streamed from.
    fn sample_gpu(measure_decode_tps: f64) -> BenchmarkSession {
        let mut sess = sample(measure_decode_tps);
        sess.engine.name = "glcuda".into();
        sess.engine.backend = "cuda".into();
        sess
    }

    /// `summary`'s roofline calls `bandwidth_for_run` directly rather than
    /// going through `analyze`, so the shared entry point is pinned on its own.
    /// A regression here mis-classifies every per-bucket verdict, not just the
    /// headline efficiency.
    #[test]
    fn the_shared_selection_picks_by_device_for_both_callers() {
        let mut cpu_run = sample(10.0);
        cpu_run.environment.hardware.cpu.read_bandwidth_gbs = Some(27.0);
        cpu_run.environment.hardware.gpu.peak_bandwidth_gbs = Some(320.0);
        assert_eq!(
            bandwidth_for_run(&cpu_run).map(|(bw, _)| bw),
            Some(27.0),
            "a CPU run is bounded by host RAM"
        );

        let mut gpu_run = sample_gpu(10.0);
        gpu_run.environment.hardware.cpu.read_bandwidth_gbs = Some(27.0);
        gpu_run.environment.hardware.gpu.peak_bandwidth_gbs = Some(320.0);
        assert_eq!(
            bandwidth_for_run(&gpu_run).map(|(bw, _)| bw),
            Some(320.0),
            "a CUDA run is bounded by device memory"
        );
    }

    /// Backend strings arrive from engine metadata, not from a Rust enum, so
    /// the match has to survive the casing an engine happens to report.
    #[test]
    fn backend_matching_is_case_insensitive() {
        let mut sess = sample(10.0);
        sess.environment.hardware.cpu.read_bandwidth_gbs = Some(27.0);
        sess.environment.hardware.gpu.peak_bandwidth_gbs = Some(320.0);
        for backend in ["CUDA", "Cuda", "cuda"] {
            sess.engine.backend = backend.into();
            assert_eq!(
                bandwidth_for_run(&sess).map(|(bw, _)| bw),
                Some(320.0),
                "backend {backend:?} should read as a GPU run"
            );
        }
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
        // This is a CPU run, so the host's measured bandwidth is the right
        // ceiling and the GPU sitting idle beside it is irrelevant. The
        // original reasoning still applies *within* the host: DDR4 channel
        // count is invisible to CPUID, so a measured host figure beats a host
        // table entry. What changed is that it no longer applies ACROSS
        // devices — see the selection comment in `analyze`.
        let mut sess = sample(10.0);
        sess.environment.hardware.cpu.read_bandwidth_gbs = Some(20.0);
        sess.environment.hardware.gpu.peak_bandwidth_gbs = Some(900.0);
        let dec = Stats::from_samples(&sess.measurements.decode_tps_samples());
        let c = analyze(&sess, &dec);
        assert_eq!(c.basis, CeilingBasis::Measured);
        assert_eq!(c.basis_bandwidth_gbs, Some(20.0));
    }

    /// **Contract change, 2026-08-20.** This previously asserted that a CPU run
    /// with no measured host bandwidth falls back to the GPU's table entry.
    /// That is the same defect as the one this module was just fixed for, only
    /// pointing the other way: a ceiling of 900 GB/s for weights streamed from
    /// host DDR would make a memory-bound run look compute-bound.
    ///
    /// No ceiling is the honest answer when the device that ran the model has
    /// no bandwidth figure. Not a weakened assertion — a corrected one.
    #[test]
    fn a_cpu_run_with_no_host_bandwidth_is_undetermined_even_beside_a_gpu() {
        let mut sess = sample(10.0);
        sess.environment.hardware.cpu.read_bandwidth_gbs = None;
        sess.environment.hardware.gpu.peak_bandwidth_gbs = Some(900.0);
        let dec = Stats::from_samples(&sess.measurements.decode_tps_samples());
        let c = analyze(&sess, &dec);
        assert_eq!(c.basis, CeilingBasis::Undetermined);
        assert_eq!(c.efficiency, None, "a GPU's bandwidth cannot bound a CPU run");
    }

    #[test]
    fn a_gpu_run_uses_the_device_table_and_says_it_is_a_spec() {
        let mut sess = sample_gpu(10.0);
        sess.environment.hardware.gpu.peak_bandwidth_gbs = Some(900.0);
        let dec = Stats::from_samples(&sess.measurements.decode_tps_samples());
        let c = analyze(&sess, &dec);
        assert_eq!(c.basis, CeilingBasis::EstimatedFromTable);
        assert_eq!(c.basis_bandwidth_gbs, Some(900.0));
        assert!(c.notes.iter().any(|n| n.contains("PUBLISHED spec, not measured")));
        assert!(c.notes.iter().any(|n| n.contains("underestimate")));
    }

    /// ⛔ Regression test for the defect a Tesla T4 run exposed on 2026-08-20.
    ///
    /// Both figures are present and the host's is *measured* while the
    /// device's comes from a table — the exact shape that made the old
    /// "measured beats table" rule pick host RAM for a CUDA run.
    #[test]
    fn a_gpu_run_never_takes_the_hosts_bandwidth_even_though_it_is_measured() {
        let mut sess = sample_gpu(10.0);
        sess.environment.hardware.cpu.read_bandwidth_gbs = Some(27.0);   // host DDR
        sess.environment.hardware.gpu.peak_bandwidth_gbs = Some(320.0);  // Tesla T4
        let dec = Stats::from_samples(&sess.measurements.decode_tps_samples());
        let c = analyze(&sess, &dec);

        assert_eq!(
            c.basis_bandwidth_gbs,
            Some(320.0),
            "a CUDA run must be bounded by device memory, not host RAM"
        );
        assert_eq!(c.basis, CeilingBasis::EstimatedFromTable);
    }

    /// ⛔ The second half of the same defect: an impossible ratio used to be
    /// clamped to exactly 1.0 and printed as "100% of peak".
    #[test]
    fn an_impossible_efficiency_is_refused_rather_than_clamped_to_look_perfect() {
        // 1 GB of weights over 27 GB/s is a 27 tok/s ceiling; claim 210.9.
        let mut sess = sample(210.9);
        sess.environment.hardware.cpu.read_bandwidth_gbs = Some(27.0);
        let dec = Stats::from_samples(&sess.measurements.decode_tps_samples());
        let c = analyze(&sess, &dec);

        assert_eq!(
            c.efficiency, None,
            "an unusable number is not a number; it must not read as 100%"
        );
        let note = c.notes.iter().find(|n| n.contains("IMPLAUSIBLE"))
            .expect("the impossibility must be stated, not hidden");
        assert!(note.contains("781%"), "the real ratio belongs in the note: {note}");
        assert!(note.contains("ceiling is wrong"), "got {note}");
    }

    /// The whole point of refusing the efficiency: nothing downstream may draw
    /// a conclusion from a ceiling that cannot be right.
    #[test]
    fn an_impossible_ceiling_yields_no_bottleneck_verdict() {
        use crate::analysis::bottleneck;

        let mut sess = sample(210.9);
        sess.environment.hardware.cpu.read_bandwidth_gbs = Some(27.0);
        let dec = Stats::from_samples(&sess.measurements.decode_tps_samples());
        let c = analyze(&sess, &dec);

        // With no efficiency the classifier falls through to phase balance and
        // declines to guess, instead of reading 781% as ">= 85% of peak" and
        // announcing `MemoryBound` with a recommendation to quantise harder.
        assert_eq!(
            bottleneck::classify(&sess, &c),
            bottleneck::Bottleneck::Undetermined,
            "a bogus ceiling must not produce a confident bottleneck verdict"
        );
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
