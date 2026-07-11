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
        s.push_str(&format!("device {cpu} ({} cores)\n", hw.cpu.logical_cores));
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
