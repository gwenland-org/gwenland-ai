//! Terminal rendering of sessions and comparisons — the default CLI output.
//!
//! Consumes a [`BenchmarkSession`] (or [`ComparisonReport`]) and produces a
//! compact, aligned text report. Every renderer in glbench reads the shared data
//! model; none computes its own facts.

use crate::comparison::runs::ComparisonReport;
use crate::comparison::statistics::Stats;
use crate::core::session::BenchmarkSession;
use crate::measurement::memory::bytes_to_gib;
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
    if let Some(bytes) = hw.storage.model_file_bytes {
        s.push_str(&format!("weights {:.2} GiB\n", bytes_to_gib(bytes)));
    }
    s.push('\n');

    // Throughput table.
    let mut t = Table::new(&["phase", "mean", "median", "min", "max", "p95", "std"])
        .right_align(1)
        .right_align(2)
        .right_align(3)
        .right_align(4)
        .right_align(5)
        .right_align(6);
    t.row(&stat_cells("prefill", &pre));
    t.row(&stat_cells("decode", &dec));
    s.push_str(&t.render());
    s.push('\n');

    if let Some(a) = &session.analysis {
        s.push_str(&format!(
            "health {:.0}%  |  bottleneck: {}",
            a.health * 100.0,
            a.bottleneck.as_str()
        ));
        if let Some(eff) = a.ceiling_efficiency {
            s.push_str(&format!("  |  {:.0}% of ceiling", eff * 100.0));
        }
        s.push_str("\n\n");
        for note in &a.notes {
            s.push_str(&format!("  - {note}\n"));
        }
    }

    if let Some(t) = &session.telemetry {
        s.push_str(&telemetry(t, hw.cpu.read_bandwidth_gbs));
    }

    if let Some(b) = &session.behavior {
        s.push_str(&behavior(b));
    }

    if let Some(v) = &session.validation {
        if !v.passed() || !v.findings.is_empty() {
            s.push_str(&format!(
                "\nvalidation: {}\n",
                if v.passed() { "passed (with notes)" } else { "FAILED" }
            ));
            for f in &v.findings {
                s.push_str(&format!("  [{}] {}: {}\n", f.severity.as_str(), f.check, f.message));
            }
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
        for st in p.hotspots() {
            let ceil_cell = match (ceiling_gbs, st.gb_per_s()) {
                (Some(c), Some(_)) => match st.ceiling_frac(c) {
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
                "  ceiling {c:.1} GB/s (measured)  |  '!' = under 25% of it, so NOT bandwidth-bound\n"
            ));
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

/// The behavior sections: what the model did, in pure numbers.
///
/// Every line is a measurement. None of it is a verdict — where a threshold is
/// applied (`degenerate`, `stalls`), it is flagged as a hint and the raw number
/// is printed next to it so the reader can disagree.
fn behavior(b: &crate::behavior::BehaviorReport) -> String {
    let mut s = String::new();
    s.push_str("\nbehavior (from a separate traced run — tracing perturbs timing)\n");

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
        s.push_str(&format!(
            "  entropy      mean {:.2} nats | p95 {:.2} | top-prob {:.2}\n",
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

fn stat_cells(label: &str, s: &Stats) -> Vec<String> {
    vec![
        label.to_string(),
        format!("{:.1}", s.mean),
        format!("{:.1}", s.median),
        format!("{:.1}", s.min),
        format!("{:.1}", s.max),
        format!("{:.1}", s.p95),
        format!("{:.1}", s.std_dev),
    ]
}
