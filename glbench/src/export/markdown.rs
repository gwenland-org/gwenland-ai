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
    let hw = &session.environment.hardware;
    if let Some(name) = &hw.gpu.name {
        s.push_str(&format!(
            "- **Device:** {} ({})\n",
            name,
            hw.gpu.compute.as_deref().unwrap_or("?")
        ));
    } else if let Some(cpu) = &hw.cpu.model {
        s.push_str(&format!("- **Device:** {cpu} ({} cores)\n", hw.cpu.logical_cores));
    }
    s.push_str(&format!(
        "- **Iterations:** {} warmup + {} measured\n\n",
        session.workload.warmup_iters, dec.count
    ));

    // Throughput table.
    s.push_str("## Throughput (tokens/second)\n\n");
    s.push_str("| Phase | mean | median | min | max | p95 | std |\n");
    s.push_str("|-------|-----:|-------:|----:|----:|----:|----:|\n");
    s.push_str(&stat_row("prefill", &pre));
    s.push_str(&stat_row("decode", &dec));
    s.push('\n');

    // Analysis.
    if let Some(a) = &session.analysis {
        s.push_str("## Analysis\n\n");
        s.push_str(&format!("- **Health:** {:.0}%\n", a.health * 100.0));
        s.push_str(&format!("- **Bottleneck:** {}\n", a.bottleneck.as_str()));
        if let Some(eff) = a.ceiling_efficiency {
            s.push_str(&format!("- **Ceiling efficiency:** {:.0}%\n", eff * 100.0));
        }
        if !a.notes.is_empty() {
            s.push_str("\n**Observations:**\n\n");
            for note in &a.notes {
                s.push_str(&format!("- {note}\n"));
            }
        }
        s.push('\n');
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
    format!(
        "| {label} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} |\n",
        s.mean, s.median, s.min, s.max, s.p95, s.std_dev
    )
}
