//! Terminal rendering of sessions and comparisons — the default CLI output.
//!
//! Consumes a [`BenchmarkSession`] (or [`ComparisonReport`]) and produces a
//! compact, aligned text report. Every renderer in glbench reads the shared data
//! model; none computes its own facts.

use crate::comparison::runs::ComparisonReport;
use crate::comparison::statistics::Stats;
use crate::core::session::BenchmarkSession;
use crate::measurement::memory::bytes_to_gib;
use crate::environment::bandwidth::CEILING_TOLERANCE;
use crate::render::table::Table;

/// Render a full session report for the terminal.
pub fn session(session: &BenchmarkSession) -> String {
    let mut s = String::new();
    let m = &session.measurements;
    let dec = Stats::from_samples(&m.decode_tps_samples());
    let pre = Stats::from_samples(&m.prefill_tps_samples());

    s.push_str(&format!("glbench :: {}\n", session.metadata.label));
    s.push_str(&format!(
        "engine {} ({}) | model {}\n",
        session.engine.name,
        session.engine.backend,
        session.workload.model_path,
    ));
    s.push_str(&format!(
        "run at unix {} | {} {} | glbench {}\n",
        session.metadata.created_unix,
        session.environment.runtime.os,
        session.environment.runtime.arch,
        session.environment.runtime.glbench_version,
    ));
    let hw = &session.environment.hardware;
    if let Some(name) = &hw.gpu.name {
        s.push_str(&format!(
            "device {} ({})",
            name,
            hw.gpu.compute.as_deref().unwrap_or("?")
        ));
        if let Some(bw) = hw.gpu.peak_bandwidth_gbs {
            s.push_str(&format!(" | peak {bw:.0} GB/s"));
        }
        s.push('\n');
    } else if let Some(cpu) = &hw.cpu.model {
        s.push_str(&format!("device {cpu} ("));
        match hw.cpu.physical_cores {
            Some(p) => s.push_str(&format!("{p}p/{}l cores", hw.cpu.logical_cores)),
            None => s.push_str(&format!("{} cores", hw.cpu.logical_cores)),
        }
        s.push_str(")\n");
    }
    // Capability: what the CPU supports. The engine's actual pick is printed in
    // the profile section below — they differ, and the difference is the point.
    let isa = hw.cpu.isa.names();
    if !isa.is_empty() {
        s.push_str(&format!("isa {}\n", isa.join(" ")));
    }
    match (hw.memory.total_bytes, hw.memory.available_bytes) {
        (Some(total), Some(avail)) => s.push_str(&format!(
            "ram {:.1} GiB total, {:.1} GiB available\n",
            bytes_to_gib(total),
            bytes_to_gib(avail),
        )),
        (Some(total), None) => s.push_str(&format!("ram {:.1} GiB total\n", bytes_to_gib(total))),
        (None, _) => s.push_str("ram not available (no /proc/meminfo on this OS)\n"),
    }
    match hw.thermal.avg_mhz {
        Some(avg) => {
            s.push_str(&format!("clock avg {avg:.0} MHz during run"));
            if let (Some(start), Some(end)) = (hw.thermal.start_mhz, hw.thermal.end_mhz) {
                s.push_str(&format!(" (start {start:.0}, end {end:.0}"));
                if hw.thermal.throttled() {
                    s.push_str(", THROTTLED");
                }
                s.push(')');
            }
            s.push('\n');
        }
        None => s.push_str("clock avg not available (no readable clock speed on this OS)\n"),
    }
    if let Some(bytes) = hw.storage.model_file_bytes {
        s.push_str(&format!("weights {:.2} GiB\n", bytes_to_gib(bytes)));
    }
    s.push('\n');

    // Throughput table.
    let mut t = Table::new(&["phase", "mean", "median", "min", "max", "p95", "std", "±95% CI"])
        .right_align(1)
        .right_align(2)
        .right_align(3)
        .right_align(4)
        .right_align(5)
        .right_align(6)
        .right_align(7);
    t.row(&stat_cells("prefill", &pre));
    t.row(&stat_cells("decode", &dec));
    s.push_str(&t.render());
    // Cold-start iterations, kept out of the warm statistics: they are what
    // a deployment's first few requests pay. Reported as median + range
    // rather than one sample — "always ~500ms" and "usually 90ms but once
    // 900ms" both average to the same number, and only the range tells them
    // apart (the same reasoning `stall_count` uses for inter-token gaps).
    if !m.cold.is_empty() {
        let cold_pre = Stats::from_samples(&m.cold_prefill_tps_samples());
        let cold_dec = Stats::from_samples(&m.cold_decode_tps_samples());
        s.push_str(&format!(
            "cold start ({} iters): prefill median {:.1} tok/s (range {:.1}-{:.1}) | decode median {:.1} tok/s (range {:.1}-{:.1}) | warm median {:.1} / {:.1}\n",
            m.cold.len(),
            cold_pre.median, cold_pre.min, cold_pre.max,
            cold_dec.median, cold_dec.min, cold_dec.max,
            pre.median,
            dec.median,
        ));
    }
    match m.joules_per_token() {
        Some(jpt) => s.push_str(&format!(
            "energy: {:.2} J/token ({:.1} J total, RAPL package-level)\n",
            jpt,
            m.energy_joules.unwrap_or(0.0),
        )),
        // `None` here is unambiguous: RAPL is Linux-only and the meter never
        // estimates from TDP, so absence always means "not available", never
        // "zero". Say so — a missing line reads as "forgot to check", not as
        // "not applicable on this OS".
        None => s.push_str("energy: not available (RAPL is Linux-only; not estimated from TDP)\n"),
    }
    s.push('\n');

    if let Some(a) = &session.analysis {
        s.push_str(&format!(
            "health {:.0}%  |  bottleneck: {}",
            a.health * 100.0,
            a.bottleneck.as_str()
        ));
        if let Some(eff) = a.ceiling_efficiency {
            let basis_tag = match a.ceiling_basis {
                crate::analysis::ceiling::CeilingBasis::Measured => "",
                crate::analysis::ceiling::CeilingBasis::EstimatedFromTable => " (est.)",
                crate::analysis::ceiling::CeilingBasis::Undetermined => "",
            };
            s.push_str(&format!("  |  {:.0}% of ceiling{basis_tag}", eff * 100.0));
        }
        s.push_str("\n\n");
        for note in &a.notes {
            s.push_str(&format!("  - {note}\n"));
        }

        if let Some(r) = &a.roofline {
            s.push_str(&roofline(r));
        }

        if !a.hypotheses.is_empty() {
            s.push_str("\nhypotheses (patterns, not verdicts — each says what the data is consistent with)\n");
            for h in &a.hypotheses {
                s.push_str(&format!("  * {h}\n"));
            }
        }
    }

    if let Some(t) = &session.telemetry {
        s.push_str(&telemetry(t, hw.cpu.read_bandwidth_gbs));
    }

    if let Some(b) = &session.behavior {
        s.push_str(&behavior(b, uses_synthetic_prompts(session)));
    }

    if let Some(v) = &session.validation {
        // Always printed, including the all-clear case: an absent validation
        // line reads as "nobody checked", not as "checked and clean" — those
        // are different facts and only one of them is what a silent omission
        // would actually mean.
        s.push_str(&format!(
            "\nvalidation: {}\n",
            match (v.passed(), v.findings.is_empty()) {
                (true, true) => "passed (no findings)".to_string(),
                (true, false) => "passed (with notes)".to_string(),
                (false, _) => "FAILED".to_string(),
            }
        ));
        for f in &v.findings {
            s.push_str(&format!("  [{}] {}: {}\n", f.severity.as_str(), f.check, f.message));
        }
    }
    s
}

/// The per-bucket roofline: which part of the model sits where relative to
/// the bandwidth ceiling. Decode first, same reasoning as the timeline.
fn roofline(r: &crate::analysis::roofline::RooflineReport) -> String {
    let mut s = String::new();
    s.push_str("\nroofline");
    match r.ceiling_gbs {
        Some(c) => s.push_str(&format!(" (vs {c:.1} GB/s ceiling)\n")),
        None => s.push_str(" (no bandwidth ceiling — verdicts unknown)\n"),
    }
    for (label, buckets) in [("decode", &r.decode), ("prefill", &r.prefill)] {
        if buckets.is_empty() {
            continue;
        }
        s.push_str(&format!("  {label}:\n"));
        for b in buckets {
            let ceil = b
                .ceiling_frac
                .map(|f| format!("{:>3.0}% ceiling", f * 100.0))
                .unwrap_or_else(|| "  - ceiling".into());
            let intensity = b
                .intensity_flop_per_byte
                .map(|i| format!("{i:.2} FLOP/B"))
                .unwrap_or_else(|| "-".into());
            s.push_str(&format!(
                "    {:<10} {ceil}  {intensity:>12}  -> {}\n",
                b.bucket.as_str(),
                b.verdict.as_str(),
            ));
        }
    }
    s
}

/// The profile sections: what the engine chose, where the time went, where the
/// memory went, and how routing behaved.
///
/// Only sections the engine actually reported are printed. A missing section
/// means "not measured" — it is left out rather than rendered as zeros, because
/// a zeroed row is a claim and an absent row is an admission.
fn telemetry(t: &glcore::telemetry::EngineTelemetry, ceiling_gbs: Option<f64>) -> String {
    let mut s = String::new();

    if let Some(b) = &t.backend {
        s.push_str(&format!(
            "\nbackend: simd {} | {} threads\n",
            b.simd_path, b.threads
        ));
        for (role, kernel) in &b.kernels {
            s.push_str(&format!("  {role:<20} {kernel}\n"));
        }
    }

    // Timeline. Decode first: it is the phase that dominates a real session,
    // and the one whose hotspot decides what to optimize next.
    for (label, phase) in [("decode", &t.decode), ("prefill", &t.prefill)] {
        let Some(p) = phase else { continue };
        if p.total_ms <= 0.0 {
            continue;
        }
        s.push_str(&format!("\n{label} timeline ({:.1} ms total)\n", p.total_ms));
        let mut tab = Table::new(&[
            "stage", "ms", "share", "ms/call", "GB/s", "%ceil", "GMAC/s",
        ])
        .right_align(1)
        .right_align(2)
        .right_align(3)
        .right_align(4)
        .right_align(5)
        .right_align(6);
        let mut over_ceiling = false;
        for st in p.hotspots() {
            let ceil_cell = match (ceiling_gbs, st.gb_per_s()) {
                (Some(c), Some(_)) => match st.ceiling_frac(c) {
                    // Nothing can read faster than DRAM allows. A stage above
                    // 100% is not an impossible stage — it is a ceiling that was
                    // measured while the machine was slower than it was during
                    // the stage (thermal state, contention). Say so, rather than
                    // print "101%" as if it were an efficiency.
                    Some(f) if f > CEILING_TOLERANCE => {
                        over_ceiling = true;
                        format!("{:.0}% ?", f * 100.0)
                    }
                    // Flag stages far from the ceiling: they are NOT
                    // bandwidth-bound, so reading fewer bytes will not speed
                    // them up. Mistaking one for the other is exactly how the
                    // native-Q4_K experiment lost 33%.
                    Some(f) if f < 0.25 => format!("{:.0}% !", f * 100.0),
                    Some(f) => format!("{:.0}%", f * 100.0),
                    None => "-".into(),
                },
                _ => "-".into(),
            };
            tab.row(&[
                st.name.clone(),
                format!("{:.2}", st.total_ms),
                match st.share_of(p.total_ms) {
                    Some(f) => format!("{:.1}%", f * 100.0),
                    None => "-".into(),
                },
                // Cost of one invocation. `share` says where the time went;
                // `ms/call` says whether a stage is slow or merely frequent.
                if st.calls > 0 {
                    format!("{:.3}", st.total_ms / st.calls as f64)
                } else {
                    "-".into()
                },
                st.gb_per_s().map_or("-".into(), |v| format!("{v:.1}")),
                ceil_cell,
                // The number that actually diagnoses a kernel. GB/s cannot
                // compare formats (different bytes per MAC); GMAC/s can.
                st.gmac_per_s().map_or("-".into(), |v| format!("{v:.1}")),
            ]);
        }
        s.push_str(&tab.render());

        if let Some(c) = ceiling_gbs {
            s.push_str(&format!(
                "  ceiling {c:.1} GB/s (measured)  |  '!' = under 25%, NOT bandwidth-bound\n"
            ));
            if over_ceiling {
                // The stage is not impossible; the ruler is. Saying which one is
                // wrong is the whole difference between a useful report and a
                // misleading one.
                s.push_str(
                    "  '?' = above the measured ceiling, which means the CEILING is wrong, \
                     not the stage:\n       it was measured while the machine was slower \
                     (thermal / contention). Treat as ~100%.\n",
                );
            }
        }

        let un = p.unattributed_ms();
        if un > 0.0 {
            // Surfaced, not hidden: a large residual means the engine's
            // instrumentation has a blind spot, which is worth knowing.
            s.push_str(&format!(
                "  unattributed {:.2} ms ({:.1}%)\n",
                un,
                un / p.total_ms * 100.0
            ));
        }
    }

    if let Some(m) = &t.memory {
        s.push_str(&format!(
            "\nmemory: model {:.2} GiB | kv cache {:.2} GiB\n",
            bytes_to_gib(m.model_bytes),
            bytes_to_gib(m.kv_cache_bytes),
        ));
    }

    if let Some(m) = &t.moe {
        s.push_str(&format!(
            "\nmoe: {} experts, top-{} | {} routed layers\n",
            m.num_experts, m.num_experts_per_tok, m.moe_layers
        ));
        s.push_str(&format!(
            "  experts touched {}/{}\n",
            m.experts_touched(),
            m.num_experts
        ));
        if let Some((min, max, mean)) = m.load_balance() {
            s.push_str(&format!(
                "  load per live expert  min {min} | max {max} | mean {mean:.1}\n"
            ));
        }
        if let Some(e) = m.routing_entropy() {
            // The one number that says whether routing is healthy. A collapsing
            // router shows up here long before output quality degrades.
            s.push_str(&format!(
                "  routing entropy {e:.3} (1.0 = uniform, 0.0 = collapsed)\n"
            ));
        }
    }

    s
}

/// Whether `session` ran against synthetic (not real-prompt) token ids —
/// true exactly for the `gllm` engine, which has no tokenizer yet and
/// generates a deterministic filler sequence instead of encoding
/// `--prompt` (see `glictus_caliburni::runtime::GllmEngine`'s docs and
/// `engine::adapter::synthetic_token_ids`). Behavior signals computed over
/// that sequence describe the model's reaction to filler, not to the prompt
/// text a reader might assume was used.
fn uses_synthetic_prompts(session: &BenchmarkSession) -> bool {
    session.workload.engine == "gllm"
}

/// The behavior sections: what the model did, in pure numbers.
///
/// Every line is a measurement. None of it is a verdict — where a threshold is
/// applied (`degenerate`, `stalls`), it is flagged as a hint and the raw number
/// is printed next to it so the reader can disagree.
fn behavior(b: &crate::behavior::BehaviorReport, synthetic_prompt: bool) -> String {
    let mut s = String::new();
    s.push_str("\nbehavior (from a separate traced run — tracing perturbs timing)");
    if synthetic_prompt {
        s.push_str(
            "\n  *** FOR SYNTHETIC PROMPT ONLY *** — gllm has no tokenizer yet; these numbers \
             describe the model's reaction to a deterministic filler sequence, not to any \
             real prompt text.",
        );
    }
    s.push('\n');

    if let Some(r) = &b.repetition {
        s.push_str(&format!(
            "  repetition   1-gram {:.2} | 2-gram {:.2} | 3-gram {:.2} | max run {}{}\n",
            r.unique_1gram_ratio,
            r.unique_2gram_ratio,
            r.unique_3gram_ratio,
            r.max_token_run,
            if r.looks_degenerate() { "  <- LOOPING" } else { "" },
        ));
    }
    if let Some(e) = &b.entropy {
        // The CoT-aware read rides on the entropy line: same number, opposite
        // meaning depending on whether the model has a thinking mode.
        let flag = match b.cot.as_ref().map(|c| c.flag) {
            Some(crate::behavior::cot::EntropyFlag::LowEntropyCotExpected) => {
                "  [COT_EXPECTED — thinking model]"
            }
            Some(crate::behavior::cot::EntropyFlag::LowEntropyAnomaly) => {
                "  <- LOW_ENTROPY_ANOMALY"
            }
            _ => "",
        };
        s.push_str(&format!(
            "  entropy      mean {:.2} nats | p95 {:.2} | top-prob {:.2}{flag}\n",
            e.mean, e.p95, e.mean_top_prob
        ));
    }
    if let Some(o) = &b.ood {
        s.push_str(&format!(
            "  perplexity   {:.1} | worst-token surprise {:.1} nats\n",
            o.perplexity, o.p95_surprise
        ));
    }
    if let Some(h) = &b.hallucination {
        // Named honestly: this is confidence/rank divergence, NOT hallucination
        // detection. See behavior::hallucination — a confidently-wrong model
        // scores clean here.
        s.push_str(&format!(
            "  confidence   top-choice {:.0}% | mean rank {:.1} | uncertain off-pick {:.0}%\n",
            h.top_choice_rate * 100.0,
            h.mean_rank,
            h.uncertain_offpick_rate * 100.0,
        ));
    }
    if let Some(st) = &b.stall {
        s.push_str(&format!(
            "  stall        p50 {:.1} ms | p99 {:.1} ms | max {:.1} ms | jitter {:.2}{}\n",
            st.p50_ms,
            st.p99_ms,
            st.max_ms,
            st.jitter,
            if st.has_stalls() {
                format!("  <- {} SPIKE(S)", st.stall_count)
            } else {
                String::new()
            },
        ));
    }
    if let Some(a) = &b.anomaly {
        s.push_str(&format!(
            "  drift        quarters {:.1} / {:.1} / {:.1} / {:.1} ms | Δ {:+.0}%{}\n",
            a.quarter_gap_ms[0],
            a.quarter_gap_ms[1],
            a.quarter_gap_ms[2],
            a.quarter_gap_ms[3],
            a.drift_frac * 100.0,
            if a.has_drift() { "  <- DRIFT" } else { "" },
        ));
        if let (Some(r), Some(at)) = (a.spike_ratio, a.spike_token) {
            s.push_str(&format!(
                "  ood window   worst {:.1}x baseline perplexity at token {}\n",
                r, at
            ));
        }
    }
    s
}

/// Render a comparison report for the terminal.
pub fn comparison(c: &ComparisonReport) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "glbench compare :: {} (baseline) vs {} (candidate)\n\n",
        c.baseline_label, c.candidate_label
    ));

    let mut t = Table::new(&["metric", "baseline", "candidate", "change", "ratio"])
        .right_align(1)
        .right_align(2)
        .right_align(3)
        .right_align(4);
    t.row(&[
        "decode tps".into(),
        format!("{:.1}", c.decode_tps.baseline),
        format!("{:.1}", c.decode_tps.candidate),
        format!("{:+.1}%", c.decode_tps.relative() * 100.0),
        format!("{:.2}x", c.decode_tps.ratio()),
    ]);
    t.row(&[
        "prefill tps".into(),
        format!("{:.1}", c.prefill_tps.baseline),
        format!("{:.1}", c.prefill_tps.candidate),
        format!("{:+.1}%", c.prefill_tps.relative() * 100.0),
        format!("{:.2}x", c.prefill_tps.ratio()),
    ]);
    s.push_str(&t.render());
    s.push_str(&format!("\nverdict: {}\n", c.regression.as_str()));
    for note in &c.notes {
        s.push_str(&format!("  - {note}\n"));
    }
    s
}

/// Render a numerical-parity report for the terminal.
pub fn parity(r: &crate::validation::parity::ParityReport) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "glbench validate :: {} vs oracle {} | model {}\n\n",
        r.candidate_engine, r.oracle_engine, r.model_path
    ));
    s.push_str(&format!(
        "matching prefix: {}/{} tokens ({:.1}% agreement)\n",
        r.check.matching_prefix,
        r.check.compared,
        r.check.agreement() * 100.0,
    ));
    s.push_str(&format!(
        "verdict: {}\n",
        if r.passed() {
            "PASS — exact match under greedy decoding"
        } else {
            "FAIL — candidate diverges from the oracle"
        }
    ));
    s
}

/// Render a scaling sweep report for the terminal.
pub fn sweep(r: &crate::runner::scale::SweepReport) -> String {
    let mut s = String::new();
    s.push_str("glbench scale :: decode throughput vs. token budget\n\n");

    let mut t = Table::new(&["tokens", "decode tps (mean)", "median", "std"])
        .right_align(1)
        .right_align(2)
        .right_align(3);
    for p in &r.points {
        let dec = Stats::from_samples(&p.session.measurements.decode_tps_samples());
        t.row(&[
            format!("{:.0}", p.axis),
            format!("{:.1}", dec.mean),
            format!("{:.1}", dec.median),
            format!("{:.1}", dec.std_dev),
        ]);
    }
    s.push_str(&t.render());
    s.push_str(&format!("\nverdict: {}\n", r.scaling.as_str()));
    s
}

fn stat_cells(label: &str, s: &Stats) -> Vec<String> {
    vec![
        label.to_string(),
        format!("{:.1}", s.mean),
        format!("{:.1}", s.median),
        format!("{:.1}", s.min),
        format!("{:.1}", s.max),
        format!("{:.1}", s.p95),
        format!("{:.1}", s.std_dev),
        // Below 3 samples a t-interval has 0-1 degrees of freedom and is not
        // a claim worth making — say so rather than leaving the column blank
        // (which would read as "0", not as "not computed").
        match s.ci95 {
            Some(ci) => format!("{ci:.1}"),
            None => "n/a".to_string(),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::metrics::{IterationMetrics, MeasurementSet};
    use crate::core::result::SessionMetadata;
    use crate::core::workload::WorkloadSpec;
    use crate::engine::metadata::EngineMetadata;
    use crate::environment::hardware::EnvironmentSnapshot;
    use crate::validation::integrity::{Severity, ValidationReport};

    fn sample() -> BenchmarkSession {
        let mut m = MeasurementSet::default();
        m.iterations.push(IterationMetrics {
            prompt_tokens: 100,
            generated_tokens: 128,
            prefill_ms: 100.0,
            decode_ms: 4000.0,
            total_ms: 4100.0,
        });
        BenchmarkSession::new(
            SessionMetadata::new("test-run"),
            EnvironmentSnapshot::probe(""),
            EngineMetadata {
                name: "glproc".into(),
                backend: "cpu".into(),
                available: true,
                model_arch: Some("qwen2".into()),
                quantization: Some("Q8_0".into()),
                thinking_capable: Some(false),
            },
            WorkloadSpec::default(),
            m,
        )
    }

    #[test]
    fn session_report_always_shows_os_arch_and_timestamp() {
        let s = session(&sample());
        assert!(s.contains(&crate::core::schema::GLBENCH_VERSION.to_string()));
        // RuntimeInfo::probe() always fills os/arch from std::env::consts, so
        // these are never empty on any platform this crate builds on.
        assert!(!std::env::consts::OS.is_empty());
        assert!(s.contains(std::env::consts::OS));
        assert!(s.contains(std::env::consts::ARCH));
        assert!(s.contains("run at unix"));
    }

    #[test]
    fn energy_prints_not_available_rather_than_being_omitted() {
        // energy_joules is None on every machine without readable RAPL
        // (every non-Linux CI runner, and most Linux ones without
        // permissions) — force it explicitly so the assertion holds
        // regardless of what happens to probe true here.
        let mut sess = sample();
        sess.measurements.energy_joules = None;
        let s = session(&sess);
        assert!(s.contains("energy: not available"), "{s}");
    }

    #[test]
    fn validation_prints_even_when_it_all_passed_with_no_findings() {
        let mut sess = sample();
        sess.validation = Some(ValidationReport::default());
        let s = session(&sess);
        assert!(s.contains("validation: passed (no findings)"), "{s}");
    }

    #[test]
    fn validation_distinguishes_passed_with_notes_from_clean() {
        let mut sess = sample();
        let mut v = ValidationReport::default();
        v.push(Severity::Warning, "integrity", "noisy run");
        sess.validation = Some(v);
        let s = session(&sess);
        assert!(s.contains("validation: passed (with notes)"), "{s}");
        assert!(s.contains("noisy run"), "{s}");
    }

    #[test]
    fn gllm_behavior_section_is_labeled_synthetic() {
        let mut sess = sample();
        sess.workload.engine = "gllm".to_string();
        sess.behavior = Some(crate::behavior::BehaviorReport {
            repetition: crate::behavior::repetition::RepetitionSignal::compute(&[1, 2, 3]),
            entropy: None,
            stall: None,
            ood: None,
            hallucination: None,
            anomaly: None,
            cot: None,
        });
        let s = session(&sess);
        assert!(s.contains("FOR SYNTHETIC PROMPT ONLY"), "{s}");
    }

    #[test]
    fn non_gllm_behavior_section_is_not_labeled_synthetic() {
        let mut sess = sample();
        sess.workload.engine = "glproc".to_string();
        sess.behavior = Some(crate::behavior::BehaviorReport {
            repetition: crate::behavior::repetition::RepetitionSignal::compute(&[1, 2, 3]),
            entropy: None,
            stall: None,
            ood: None,
            hallucination: None,
            anomaly: None,
            cot: None,
        });
        let s = session(&sess);
        assert!(!s.contains("SYNTHETIC PROMPT"), "{s}");
    }

    #[test]
    fn avg_clock_prints_the_reading_and_throttle_flag() {
        let mut sess = sample();
        sess.environment.hardware.thermal = crate::environment::thermal::ThermalSnapshot {
            start_mhz: Some(3000.0),
            end_mhz: Some(2500.0),
            avg_mhz: Some(2750.0),
        };
        let s = session(&sess);
        assert!(s.contains("clock avg 2750 MHz"), "{s}");
        assert!(s.contains("THROTTLED"), "{s}");
    }

    #[test]
    fn no_clock_reading_says_not_available() {
        let mut sess = sample();
        sess.environment.hardware.thermal = crate::environment::thermal::ThermalSnapshot::default();
        let s = session(&sess);
        assert!(s.contains("clock avg not available"), "{s}");
    }
}
