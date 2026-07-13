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
        s.push_str(&telemetry(t));
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
fn telemetry(t: &glcore::telemetry::EngineTelemetry) -> String {
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
        let mut tab = Table::new(&["stage", "ms", "share", "calls", "ms/call"])
            .right_align(1)
            .right_align(2)
            .right_align(3)
            .right_align(4);
        for st in p.hotspots() {
            tab.row(&[
                st.name.clone(),
                format!("{:.2}", st.total_ms),
                match st.share_of(p.total_ms) {
                    Some(f) => format!("{:.1}%", f * 100.0),
                    None => "-".into(),
                },
                st.calls.to_string(),
                // Cost of one invocation — the number to act on. `share` says
                // where the time went; `ms/call` says whether a stage is slow
                // or merely frequent, and those call for different fixes.
                if st.calls > 0 {
                    format!("{:.3}", st.total_ms / st.calls as f64)
                } else {
                    "-".into()
                },
            ]);
        }
        s.push_str(&tab.render());
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
