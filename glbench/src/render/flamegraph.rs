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

/// Render a training step's phase breakdown as the same proportional bars.
///
/// Deliberately takes the four durations rather than a `VLTrainingAttribution`,
/// so this stays ungated and testable in a default build — the same split
/// `render::loss_curve` makes. The caller has the struct; this only needs
/// numbers.
///
/// `total_ms` is the measured step time, not the sum of the phases. When the
/// three do not account for the whole step the remainder is drawn as
/// `unattributed`, exactly as [`render_phase`] does for inference — a
/// breakdown that silently sums to less than the whole invites the reader to
/// assume it sums to the whole.
pub fn render_training_step(
    forward_ms: f64,
    backward_ms: f64,
    optimizer_ms: f64,
    total_ms: f64,
) -> String {
    if total_ms <= 0.0 {
        return String::new();
    }
    let mut rows: Vec<(String, f64)> = vec![
        ("forward".to_string(), forward_ms),
        ("backward".to_string(), backward_ms),
        ("optimizer".to_string(), optimizer_ms),
    ];
    // Widest first, matching the inference view's ranking.
    rows.sort_by(|a, b| b.1.total_cmp(&a.1));

    let unattributed = total_ms - forward_ms - backward_ms - optimizer_ms;
    if unattributed > 0.0 {
        rows.push(("unattributed".to_string(), unattributed));
    }

    let mut s = format!("
training step flame ({total_ms:.3} ms total)
");
    for (name, ms) in rows {
        let share = (ms / total_ms).clamp(0.0, 1.0);
        let filled = ((share * BAR_WIDTH as f64).round() as usize).min(BAR_WIDTH);
        let bar: String = "#".repeat(filled) + &".".repeat(BAR_WIDTH - filled);
        s.push_str(&format!(
            "  {name:<NAME_WIDTH$} {bar} {:5.1}% {ms:>8.3} ms
",
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

    // -----------------------------------------------------------------------
    // Training step breakdown (Wave 5)
    // -----------------------------------------------------------------------

    #[test]
    fn a_training_step_is_drawn_widest_phase_first() {
        // backward dominates, then optimizer, then forward.
        let out = render_training_step(1.0, 5.0, 2.0, 8.0);
        let names: Vec<&str> = out
            .lines()
            .filter(|l| l.starts_with("  "))
            .filter_map(|l| l.split_whitespace().next())
            .collect();
        assert_eq!(names, vec!["backward", "optimizer", "forward"]);
        assert!(out.contains("62.5%"), "backward is 5/8: {out}");
    }

    #[test]
    fn a_gap_between_the_phases_and_the_total_is_drawn_as_unattributed() {
        // Phases sum to 3ms, the step took 5ms.
        let out = render_training_step(1.0, 1.0, 1.0, 5.0);
        assert!(out.contains("unattributed"), "{out}");
        assert!(out.contains("40.0%"), "the 2ms gap is 40% of 5ms: {out}");
    }

    #[test]
    fn a_fully_attributed_step_draws_no_unattributed_row() {
        let out = render_training_step(1.0, 2.0, 1.0, 4.0);
        assert!(!out.contains("unattributed"), "{out}");
    }

    #[test]
    fn a_zero_duration_step_draws_nothing_rather_than_dividing_by_zero() {
        assert!(render_training_step(0.0, 0.0, 0.0, 0.0).is_empty());
    }

    /// Clock skew could make the phases exceed the total. Clamp the bar rather
    /// than overflowing the width or printing a share above 100%.
    #[test]
    fn phases_exceeding_the_total_are_clamped_not_overflowed() {
        let out = render_training_step(3.0, 3.0, 3.0, 1.0);
        for row in out.lines().filter(|l| l.starts_with("  ")) {
            // The bar is the one whitespace-delimited token made entirely of
            // '#' and '.'; counting those characters across the whole row would
            // also pick up the decimal point in "3.000 ms".
            let bar = row
                .split_whitespace()
                .find(|t| !t.is_empty() && t.chars().all(|c| c == '#' || c == '.'))
                .unwrap_or_else(|| panic!("no bar in {row:?}"));
            assert_eq!(bar.len(), BAR_WIDTH, "bar must stay {BAR_WIDTH} wide: {row:?}");
        }
        assert!(!out.contains("unattributed"), "a negative gap is not drawn: {out}");
    }
}
