//! Markdown export of a session — a human-readable report for pasting into a
//! changelog or PR, matching the project's house style of dense tables plus a
//! short prose summary.

use crate::comparison::statistics::Stats;
use crate::core::session::BenchmarkSession;
use crate::measurement::memory::bytes_to_gib;

/// Render a session as a Markdown document.
pub fn render(session: &BenchmarkSession) -> String {
    let mut s = String::new();
    let m = &session.measurements;
    let dec = Stats::from_samples(&m.decode_tps_samples());
    let pre = Stats::from_samples(&m.prefill_tps_samples());

    s.push_str(&format!("# glbench — {}\n\n", session.metadata.label));

    // Header facts.
    s.push_str("## Run\n\n");
    s.push_str(&format!("- **Engine:** {} ({})\n", session.engine.name, session.engine.backend));
    s.push_str(&format!("- **Model:** {}\n", session.workload.model_path));
    if let Some(q) = &session.engine.quantization {
        s.push_str(&format!("- **Quantization:** {q}\n"));
    }
    if let Some(bytes) = session.environment.hardware.storage.model_file_bytes {
        s.push_str(&format!("- **Model size:** {:.2} GiB\n", bytes_to_gib(bytes)));
    }
    if let Some(bytes) = m.peak_memory_bytes {
        s.push_str(&format!(
            "- **Peak RSS:** {:.2} GiB (process high-water mark, includes model load)\n",
            bytes_to_gib(bytes)
        ));
    }
    let hw = &session.environment.hardware;
    if let Some(pct) = m.cpu_utilization_pct {
        s.push_str(&format!(
            "- **CPU utilization:** {pct:.1}% (measured phase, {} logical cores)\n",
            hw.cpu.logical_cores
        ));
    }
    if let Some(name) = &hw.gpu.name {
        s.push_str(&format!(
            "- **Device:** {} ({})\n",
            name,
            hw.gpu.compute.as_deref().unwrap_or("?")
        ));
    } else if let Some(cpu) = &hw.cpu.model {
        s.push_str(&format!("- **Device:** {cpu} ({} cores)\n", hw.cpu.logical_cores));
    }
    match (hw.memory.total_bytes, hw.memory.available_bytes) {
        (Some(total), Some(avail)) => s.push_str(&format!(
            "- **RAM:** {:.1} GiB total, {:.1} GiB available\n",
            bytes_to_gib(total),
            bytes_to_gib(avail),
        )),
        (Some(total), None) => {
            s.push_str(&format!("- **RAM:** {:.1} GiB total\n", bytes_to_gib(total)))
        }
        (None, _) => s.push_str("- **RAM:** not available (no /proc/meminfo on this OS)\n"),
    }
    s.push_str(&format!(
        "- **Environment:** {} {} | glbench {}\n",
        session.environment.runtime.os,
        session.environment.runtime.arch,
        session.environment.runtime.glbench_version,
    ));
    s.push_str(&format!("- **Run at:** unix {}\n", session.metadata.created_unix));
    s.push_str(&format!(
        "- **Iterations:** {} warmup + {} measured\n\n",
        session.workload.warmup_iters, dec.count
    ));

    // Throughput table.
    s.push_str("## Throughput (tokens/second)\n\n");
    s.push_str("| Phase | mean | median | min | max | p95 | std | ±95% CI |\n");
    s.push_str("|-------|-----:|-------:|----:|----:|----:|----:|--------:|\n");
    s.push_str(&stat_row("prefill", &pre));
    s.push_str(&stat_row("decode", &dec));
    s.push('\n');
    if pre.count < 3 || dec.count < 3 {
        s.push_str(
            "_±95% CI needs at least 3 measured iterations; below that a confidence \
             interval has too few degrees of freedom to mean anything, so it reads `n/a`.\
             _\n\n",
        );
    }

    if !session.measurements.cold.is_empty() {
        let cold_pre = Stats::from_samples(&session.measurements.cold_prefill_tps_samples());
        let cold_dec = Stats::from_samples(&session.measurements.cold_decode_tps_samples());
        s.push_str(&format!(
            "**Cold start** ({} iterations, excluded from the warm statistics above):\n",
            session.measurements.cold.len()
        ));
        s.push_str(&format!(
            "prefill median {:.1} tok/s (range {:.1}-{:.1}) · decode median {:.1} tok/s (range {:.1}-{:.1})\n\n",
            cold_pre.median, cold_pre.min, cold_pre.max,
            cold_dec.median, cold_dec.min, cold_dec.max,
        ));
    }
    match session.measurements.joules_per_token() {
        Some(jpt) => s.push_str(&format!("**Energy:** {jpt:.2} J/token (RAPL, package-level)\n\n")),
        // Unambiguous: RAPL is Linux-only and never estimated from TDP, so
        // absence always means "not available", never "zero" or "n/a here".
        None => s.push_str("**Energy:** not available (RAPL is Linux-only; not estimated from TDP)\n\n"),
    }

    // Analysis.
    if let Some(a) = &session.analysis {
        s.push_str("## Analysis\n\n");
        s.push_str(&format!("- **Health:** {:.0}%\n", a.health * 100.0));
        s.push_str(&format!("- **Bottleneck:** {}\n", a.bottleneck.as_str()));
        if let Some(eff) = a.ceiling_efficiency {
            use crate::analysis::ceiling::CeilingBasis;
            let basis_note = match a.ceiling_basis {
                CeilingBasis::Measured => " (measured on this machine)",
                CeilingBasis::EstimatedFromTable => " (estimated from the device's published spec, not measured)",
                CeilingBasis::Undetermined => "",
            };
            s.push_str(&format!("- **Ceiling efficiency:** {:.0}%{basis_note}\n", eff * 100.0));
        }
        if !a.notes.is_empty() {
            s.push_str("\n**Observations:**\n\n");
            for note in &a.notes {
                s.push_str(&format!("- {note}\n"));
            }
        }
        s.push('\n');

        // Per-bucket roofline.
        if let Some(r) = &a.roofline {
            s.push_str("## Roofline\n\n");
            if let Some(c) = r.ceiling_gbs {
                s.push_str(&format!("Bandwidth ceiling: {c:.1} GB/s\n\n"));
            }
            s.push_str("| Phase | Bucket | share | GB/s | % ceiling | FLOP/B | verdict |\n");
            s.push_str("|-------|--------|------:|-----:|----------:|-------:|---------|\n");
            for (phase, buckets) in [("decode", &r.decode), ("prefill", &r.prefill)] {
                for b in buckets {
                    s.push_str(&format!(
                        "| {phase} | {} | {} | {} | {} | {} | {} |\n",
                        b.bucket.as_str(),
                        b.share.map_or("-".into(), |v| format!("{:.1}%", v * 100.0)),
                        b.gb_per_s.map_or("-".into(), |v| format!("{v:.1}")),
                        b.ceiling_frac.map_or("-".into(), |v| format!("{:.0}%", v * 100.0)),
                        b.intensity_flop_per_byte.map_or("-".into(), |v| format!("{v:.2}")),
                        b.verdict.as_str(),
                    ));
                }
            }
            s.push('\n');
        }

        if !a.hypotheses.is_empty() {
            s.push_str("## Hypotheses\n\nPatterns, not verdicts — each states what the data is consistent with.\n\n");
            for h in &a.hypotheses {
                s.push_str(&format!("- {h}\n"));
            }
            s.push('\n');
        }
    }

    // Engine telemetry: what the engine chose, and where the time went.
    if let Some(t) = &session.telemetry {
        s.push_str(&telemetry_section(t, hw.cpu.read_bandwidth_gbs));
    }

    // Behavior signals: what the model did, in pure numbers (see glbench's
    // README "What a report contains" for what each one can honestly claim).
    if let Some(b) = &session.behavior {
        s.push_str(&behavior_section(b, session.workload.engine == "gllm"));
    }

    // Validation.
    if let Some(v) = &session.validation {
        s.push_str("## Validation\n\n");
        s.push_str(&format!("**Passed:** {}\n\n", if v.passed() { "yes" } else { "NO" }));
        for f in &v.findings {
            s.push_str(&format!("- `{}` [{}] {}\n", f.check, f.severity.as_str(), f.message));
        }
        s.push('\n');
    }

    s.push_str(&format!(
        "---\n_glbench {} · schema v{}_\n",
        session.metadata.glbench_version, session.metadata.schema_version
    ));
    s
}

fn stat_row(label: &str, s: &Stats) -> String {
    let ci = match s.ci95 {
        Some(ci) => format!("{ci:.1}"),
        None => "n/a".to_string(),
    };
    format!(
        "| {label} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {ci} |\n",
        s.mean, s.median, s.min, s.max, s.p95, s.std_dev
    )
}

/// Engine telemetry as Markdown: backend/kernel choice, per-phase stage
/// tables, memory split, MoE routing. Mirrors `render::text::telemetry` —
/// same data, same "only what was actually measured" rule, different format.
fn telemetry_section(t: &glcore::telemetry::EngineTelemetry, ceiling_gbs: Option<f64>) -> String {
    let mut s = String::new();
    s.push_str("## Engine Telemetry\n\n");

    if let Some(b) = &t.backend {
        s.push_str(&format!("- **SIMD path:** {}\n", b.simd_path));
        s.push_str(&format!("- **Threads:** {}\n", b.threads));
        if !b.kernels.is_empty() {
            s.push_str("\n| Role | Kernel |\n|------|--------|\n");
            for (role, kernel) in &b.kernels {
                s.push_str(&format!("| {role} | {kernel} |\n"));
            }
        }
        s.push('\n');
    }

    for (label, phase) in [("Decode", &t.decode), ("Prefill", &t.prefill)] {
        let Some(p) = phase else { continue };
        if p.total_ms <= 0.0 {
            continue;
        }
        s.push_str(&format!("### {label} timeline ({:.1} ms total)\n\n", p.total_ms));
        s.push_str("| Stage | ms | share | ms/call | GB/s | % ceiling | GMAC/s |\n");
        s.push_str("|-------|---:|------:|--------:|-----:|----------:|-------:|\n");
        for st in p.hotspots() {
            let ceil_cell = match (ceiling_gbs, st.gb_per_s()) {
                (Some(c), Some(_)) => {
                    st.ceiling_frac(c).map_or("-".into(), |f| format!("{:.0}%", f * 100.0))
                }
                _ => "-".into(),
            };
            s.push_str(&format!(
                "| {} | {:.2} | {} | {} | {} | {ceil_cell} | {} |\n",
                st.name,
                st.total_ms,
                st.share_of(p.total_ms).map_or("-".into(), |f| format!("{:.1}%", f * 100.0)),
                if st.calls > 0 {
                    format!("{:.3}", st.total_ms / st.calls as f64)
                } else {
                    "-".into()
                },
                st.gb_per_s().map_or("-".into(), |v| format!("{v:.1}")),
                st.gmac_per_s().map_or("-".into(), |v| format!("{v:.1}")),
            ));
        }
        let un = p.unattributed_ms();
        if un > 0.0 {
            s.push_str(&format!(
                "\nUnattributed: {:.2} ms ({:.1}%)\n",
                un,
                un / p.total_ms * 100.0
            ));
        }
        let flame = crate::render::flamegraph::render_phase(label, p);
        if !flame.is_empty() {
            s.push_str("\n```text\n");
            s.push_str(flame.trim_start_matches('\n'));
            s.push_str("```\n");
        }
        s.push('\n');
    }

    if let Some(m) = &t.memory {
        s.push_str(&format!(
            "- **Memory:** model {:.2} GiB · KV cache {:.2} GiB · scratch {:.2} GiB\n\n",
            bytes_to_gib(m.model_bytes),
            bytes_to_gib(m.kv_cache_bytes),
            bytes_to_gib(m.scratch_bytes),
        ));
    }

    if let Some(m) = &t.moe {
        s.push_str(&format!(
            "- **MoE:** {} experts, top-{} · {} routed layers · {}/{} touched\n",
            m.num_experts,
            m.num_experts_per_tok,
            m.moe_layers,
            m.experts_touched(),
            m.num_experts,
        ));
        if let Some((min, max, mean)) = m.load_balance() {
            s.push_str(&format!("- **Load per live expert:** min {min} · max {max} · mean {mean:.1}\n"));
        }
        if let Some(e) = m.routing_entropy() {
            s.push_str(&format!("- **Routing entropy:** {e:.3} (1.0 = uniform, 0.0 = collapsed)\n"));
        }
        s.push('\n');
    }

    s
}

/// Behavior signals as Markdown — mirrors `render::text::behavior`'s content
/// and honesty rules (`hallucination` named as a proxy, `toxicity` never
/// implied to exist).
///
/// `synthetic_prompt` is true exactly for the `gllm` engine (no tokenizer
/// yet, so `--prompt` is never actually encoded — see
/// `glictus_caliburni::runtime::GllmEngine`'s docs); the section is labeled
/// so these numbers are never mistaken for a real prompt's behavior.
fn behavior_section(b: &crate::behavior::BehaviorReport, synthetic_prompt: bool) -> String {
    let mut s = String::new();
    s.push_str("## Behavior\n\n_From a separate traced run — tracing perturbs timing._\n\n");
    if synthetic_prompt {
        s.push_str(
            "> **FOR SYNTHETIC PROMPT ONLY** — gllm has no tokenizer yet; these numbers \
             describe the model's reaction to a deterministic filler sequence, not to any \
             real prompt text.\n\n",
        );
    }

    if let Some(r) = &b.repetition {
        s.push_str(&format!(
            "- **Repetition:** 1-gram {:.2} · 2-gram {:.2} · 3-gram {:.2} · max run {}{}\n",
            r.unique_1gram_ratio,
            r.unique_2gram_ratio,
            r.unique_3gram_ratio,
            r.max_token_run,
            if r.looks_degenerate() { " — **looks degenerate**" } else { "" },
        ));
    }
    if let Some(e) = &b.entropy {
        let flag = match b.cot.as_ref().map(|c| c.flag) {
            Some(crate::behavior::cot::EntropyFlag::LowEntropyCotExpected) => {
                " (COT_EXPECTED — thinking model)"
            }
            Some(crate::behavior::cot::EntropyFlag::LowEntropyAnomaly) => {
                " — **LOW_ENTROPY_ANOMALY**"
            }
            _ => "",
        };
        s.push_str(&format!(
            "- **Entropy:** mean {:.2} nats · p95 {:.2} · top-prob {:.2}{flag}\n",
            e.mean, e.p95, e.mean_top_prob
        ));
    }
    if let Some(o) = &b.ood {
        s.push_str(&format!(
            "- **Perplexity:** {:.1} · worst-token surprise {:.1} nats\n",
            o.perplexity, o.p95_surprise
        ));
    }
    if let Some(h) = &b.hallucination {
        s.push_str(&format!(
            "- **Confidence/rank divergence (proxy, not a hallucination detector):** \
             top-choice {:.0}% · mean rank {:.1} · uncertain off-pick {:.0}%\n",
            h.top_choice_rate * 100.0,
            h.mean_rank,
            h.uncertain_offpick_rate * 100.0,
        ));
    }
    if let Some(st) = &b.stall {
        s.push_str(&format!(
            "- **Stall:** p50 {:.1} ms · p95 {:.1} ms · p99 {:.1} ms · max {:.1} ms · jitter {:.2}{}\n",
            st.p50_ms,
            st.p95_ms,
            st.p99_ms,
            st.max_ms,
            st.jitter,
            if st.has_stalls() {
                format!(" — **{} spike(s)**", st.stall_count)
            } else {
                String::new()
            },
        ));
    }
    if let Some(a) = &b.anomaly {
        s.push_str(&format!(
            "- **Drift:** quarters {:.1} / {:.1} / {:.1} / {:.1} ms · Δ {:+.0}%{}\n",
            a.quarter_gap_ms[0],
            a.quarter_gap_ms[1],
            a.quarter_gap_ms[2],
            a.quarter_gap_ms[3],
            a.drift_frac * 100.0,
            if a.has_drift() { " — **drift**" } else { "" },
        ));
        if let (Some(r), Some(at)) = (a.spike_ratio, a.spike_token) {
            s.push_str(&format!(
                "- **OOD window:** worst {r:.1}x baseline perplexity at token {at}\n"
            ));
        }
    }
    // toxicity is deliberately not implemented (behavior::toxicity) — never
    // printed as if it were a measured-but-empty section.
    s.push('\n');
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::behavior::repetition::RepetitionSignal;
    use crate::behavior::BehaviorReport;
    use crate::core::metrics::{IterationMetrics, MeasurementSet};
    use crate::core::result::SessionMetadata;
    use crate::core::workload::WorkloadSpec;
    use crate::engine::metadata::EngineMetadata;
    use crate::environment::hardware::EnvironmentSnapshot;

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
    fn header_shows_os_arch_ram_and_timestamp() {
        let s = render(&sample());
        assert!(s.contains(std::env::consts::OS), "{s}");
        assert!(s.contains(std::env::consts::ARCH), "{s}");
        assert!(s.contains("Run at:"), "{s}");
        // RAM is either a real figure or the explicit not-available line —
        // never silently absent.
        assert!(s.contains("**RAM:**"), "{s}");
    }

    #[test]
    fn energy_prints_not_available_rather_than_being_omitted() {
        let mut sess = sample();
        sess.measurements.energy_joules = None;
        let s = render(&sess);
        assert!(s.contains("**Energy:** not available"), "{s}");
    }

    #[test]
    fn behavior_section_is_emitted_when_present() {
        let mut sess = sample();
        sess.behavior = Some(BehaviorReport {
            repetition: Some(RepetitionSignal::compute(&[1, 2, 3, 1, 2, 3]).unwrap()),
            entropy: None,
            stall: None,
            ood: None,
            hallucination: None,
            anomaly: None,
            cot: None,
        });
        let s = render(&sess);
        assert!(s.contains("## Behavior"), "{s}");
        assert!(s.contains("Repetition"), "{s}");
    }

    #[test]
    fn telemetry_section_is_absent_without_telemetry() {
        let s = render(&sample());
        assert!(!s.contains("## Engine Telemetry"), "{s}");
    }

    #[test]
    fn gllm_behavior_section_is_labeled_synthetic() {
        let mut sess = sample();
        sess.workload.engine = "gllm".to_string();
        sess.behavior = Some(BehaviorReport {
            repetition: Some(RepetitionSignal::compute(&[1, 2, 3, 1, 2, 3]).unwrap()),
            entropy: None,
            stall: None,
            ood: None,
            hallucination: None,
            anomaly: None,
            cot: None,
        });
        let s = render(&sess);
        assert!(s.contains("FOR SYNTHETIC PROMPT ONLY"), "{s}");
    }

    #[test]
    fn non_gllm_behavior_section_is_not_labeled_synthetic() {
        let mut sess = sample();
        sess.workload.engine = "glproc".to_string();
        sess.behavior = Some(BehaviorReport {
            repetition: Some(RepetitionSignal::compute(&[1, 2, 3, 1, 2, 3]).unwrap()),
            entropy: None,
            stall: None,
            ood: None,
            hallucination: None,
            anomaly: None,
            cot: None,
        });
        let s = render(&sess);
        assert!(!s.contains("SYNTHETIC PROMPT"), "{s}");
    }
}
