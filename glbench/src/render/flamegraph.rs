//! ASCII flame graph — a proportional-width bar chart of where a phase's time
//! went, built entirely from telemetry `run` already collects.
//!
//! [Vetted per `RESEARCH_REQUIREMENTS.md`'s 8 mandatory questions]
//! 1. What problem: the existing timeline table (`render::text::telemetry`)
//!    already lists stage/ms/share as numbers, but a proportional bar makes
//!    "what dominates this phase" visible at a glance without reading a
//!    column of percentages — the entire point of a flame graph.
//! 2. Who benefits: anyone eyeballing a report for the first time, before
//!    they'd know which numeric column to scan.
//! 3. Production/research use: flame graphs (Brendan Gregg's original tool,
//!    and every profiler UI since) are the standard visualization for "where
//!    did the time go" in performance engineering.
//! 4. How calculated: no new measurement — pure re-render of
//!    `PhaseProfile.hotspots()` (already time-sorted) and `unattributed_ms()`,
//!    the same data `render::text::telemetry` already prints as a table.
//! 5. Reproducible: yes, deterministic given the same telemetry.
//! 6. Actionable: visually reinforces the same hotspot the numeric table
//!    already names — a faster read of an existing fact, not a new one.
//! 7. Lightweight: string formatting only, no sampling, no engine hook.
//! 8. Philosophy: `render`, not `measurement` or `analysis` — this presents
//!    facts already collected, concludes nothing, judges nothing.
//!
//! # Honest scoping
//!
//! This is a **single-level** bar chart, not a hierarchical flame graph in
//! the traditional call-stack sense — glbench's telemetry is flat buckets
//! (attention / ffn / lm_head), not a call tree, so there is no stack depth
//! to draw. "Flame graph" is used here in the loose, industry-common sense of
//! "proportional-width bars ranked by time," which is what the underlying
//! data actually supports; claiming stack hierarchy that isn't there would be
//! exactly the kind of overclaim Mensura Veritatis exists to avoid.

use glcore::telemetry::PhaseProfile;

const BAR_WIDTH: usize = 40;
const NAME_WIDTH: usize = 14;

/// Render one phase's stages as proportional ASCII bars, widest (most time)
/// first. Returns an empty string for a phase with no measurable time.
pub fn render_phase(label: &str, phase: &PhaseProfile) -> String {
    if phase.total_ms <= 0.0 {
        return String::new();
    }

    let mut rows: Vec<(String, f64, f64)> = phase
        .hotspots()
        .into_iter()
        .map(|st| (st.name.clone(), st.total_ms, st.share_of(phase.total_ms).unwrap_or(0.0)))
        .collect();
    let unattributed = phase.unattributed_ms();
    if unattributed > 0.0 {
        rows.push(("unattributed".to_string(), unattributed, unattributed / phase.total_ms));
    }

    let mut s = format!("\n{label} flame ({:.1} ms total)\n", phase.total_ms);
    for (name, ms, share) in rows {
        let filled = (share * BAR_WIDTH as f64).round() as usize;
        let filled = filled.min(BAR_WIDTH);
        let bar: String = "#".repeat(filled) + &".".repeat(BAR_WIDTH - filled);
        s.push_str(&format!(
            "  {name:<NAME_WIDTH$} {bar} {:5.1}% {ms:>8.2} ms\n",
            share * 100.0
        ));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use glcore::telemetry::StageTiming;

    fn stage(name: &str, ms: f64, calls: u64) -> StageTiming {
        StageTiming { name: name.to_string(), total_ms: ms, calls, bytes_read: None, macs: None }
    }

    #[test]
    fn bars_are_ordered_by_time_descending() {
        let p = PhaseProfile {
            stages: vec![stage("attn", 20.0, 10), stage("ffn", 70.0, 10), stage("lm_head", 10.0, 10)],
            total_ms: 100.0,
        };
        let out = render_phase("decode", &p);
        let ffn_pos = out.find("ffn").unwrap();
        let attn_pos = out.find("attn").unwrap();
        let lm_head_pos = out.find("lm_head").unwrap();
        assert!(ffn_pos < attn_pos && attn_pos < lm_head_pos, "{out}");
    }

    #[test]
    fn full_share_fills_the_entire_bar() {
        let p = PhaseProfile { stages: vec![stage("only", 50.0, 1)], total_ms: 50.0 };
        let out = render_phase("decode", &p);
        assert!(out.contains(&"#".repeat(BAR_WIDTH)), "{out}");
    }

    #[test]
    fn unattributed_time_gets_its_own_row() {
        let p = PhaseProfile { stages: vec![stage("attn", 60.0, 1)], total_ms: 100.0 };
        let out = render_phase("decode", &p);
        assert!(out.contains("unattributed"), "{out}");
        assert!(out.contains("40.0%"), "{out}");
    }

    #[test]
    fn zero_time_phase_renders_nothing() {
        let p = PhaseProfile { stages: vec![], total_ms: 0.0 };
        assert_eq!(render_phase("decode", &p), "");
    }

    #[test]
    fn bar_width_never_exceeds_the_budget_even_with_rounding() {
        // Three stages each just over a third: naive rounding could push the
        // sum of filled chars past BAR_WIDTH for any single row without the
        // per-row clamp.
        let p = PhaseProfile {
            stages: vec![stage("a", 34.0, 1), stage("b", 33.0, 1), stage("c", 33.0, 1)],
            total_ms: 100.0,
        };
        let out = render_phase("decode", &p);
        for line in out.lines().filter(|l| l.contains('#')) {
            let hashes = line.chars().filter(|&c| c == '#').count();
            assert!(hashes <= BAR_WIDTH, "{line}");
        }
    }
}
