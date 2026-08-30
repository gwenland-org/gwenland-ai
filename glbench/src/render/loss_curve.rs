//! ASCII loss curve — the shape of a training run, at a glance.
//!
//! # Why this is ungated
//!
//! It takes `&[(usize, f32)]` — step index and loss — not a
//! `VLTrainingSession`. Plotting a series has nothing to do with gltrain, so
//! gating it behind `train-bench` would make the plotting *math* a
//! training-only feature for no reason. Same split D-11 draws for GLBitProf:
//! the math is ungated, the source is gated.
//!
//! It also means this module is fully testable in a default build, which
//! matters — the training tests are slow because they train.
//!
//! # What the picture is allowed to claim
//!
//! A curve is a summary, and a summary that hides its own resolution invites
//! the reader to over-read it. So the axis labels carry the real loss range and
//! the real step range, and [`render`] states how many points went into how
//! many columns. A sampled run (D-19) plots the steps it archived; the caller
//! passes what it has, and the header says how many that was.
//!
//! Nothing here smooths, interpolates, or extrapolates. Each column shows the
//! **minimum** loss of the steps that fall in it, because for a descent curve
//! the question is how low it got, and a mean would blur a spike the reader
//! wants to see.

use std::fmt::Write as _;

/// Plot width in columns, excluding the axis gutter.
const WIDTH: usize = 60;

/// Plot height in rows.
const HEIGHT: usize = 14;

/// Width of the left-hand label gutter.
const GUTTER: usize = 12;

/// Render an ASCII loss curve from `(step_index, loss)` pairs.
///
/// Returns an empty string for an empty series: there is no curve, and drawing
/// an empty box would suggest there was one that happened to be flat.
pub fn render(points: &[(usize, f32)]) -> String {
    if points.is_empty() {
        return String::new();
    }

    let losses: Vec<f32> = points.iter().map(|(_, l)| *l).collect();
    let finite: Vec<f32> = losses.iter().copied().filter(|l| l.is_finite()).collect();
    if finite.is_empty() {
        // Every loss was NaN or Inf. That is a real and important observation,
        // and it is not a curve.
        return format!(
            "\nloss curve: every one of {} points is non-finite (NaN or Inf) — \
             nothing to plot\n",
            points.len()
        );
    }

    let lo = finite.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = finite.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let first_step = points[0].0;
    let last_step = points[points.len() - 1].0;

    // Each column holds the minimum of the steps that land in it. Buckets are
    // assigned by position in the series, not by step index: with D-19
    // sampling the indices are uneven, and spacing columns by index would make
    // a thinned region look like a gap in the run.
    let columns = WIDTH.min(points.len());
    let mut buckets: Vec<Option<f32>> = vec![None; columns];
    for (i, (_, loss)) in points.iter().enumerate() {
        if !loss.is_finite() {
            continue;
        }
        let col = i * columns / points.len();
        let slot = &mut buckets[col.min(columns - 1)];
        *slot = Some(match *slot {
            Some(current) => current.min(*loss),
            None => *loss,
        });
    }

    let span = hi - lo;
    let mut out = String::new();
    let _ = writeln!(
        out,
        "\nloss curve — {} points over {} columns, steps {}..{}",
        points.len(),
        columns,
        first_step,
        last_step
    );

    for row in 0..HEIGHT {
        // Row 0 is the top of the plot, which is the highest loss.
        let upper = hi - span * (row as f32 / HEIGHT as f32);
        let lower = hi - span * ((row + 1) as f32 / HEIGHT as f32);
        // The label is the value at the top of this row's band.
        let _ = write!(out, "{upper:>GUTTER$.4} |");
        for bucket in &buckets {
            let ch = match bucket {
                // A zero span means every value is identical; put the whole
                // series on the top row rather than nowhere.
                Some(_) if span <= 0.0 && row == 0 => '*',
                Some(_) if span <= 0.0 => ' ',
                // Inclusive at the bottom so the minimum lands in the last row
                // rather than falling off the plot.
                Some(v) if *v <= upper && (*v > lower || row == HEIGHT - 1) => '*',
                _ => ' ',
            };
            out.push(ch);
        }
        out.push('\n');
    }

    let _ = write!(out, "{:>GUTTER$} +", "");
    let _ = writeln!(out, "{}", "-".repeat(columns));
    let _ = write!(out, "{:>GUTTER$}  ", "");
    let _ = writeln!(out, "step {first_step}{:>width$}", last_step, width = columns.saturating_sub(6 + first_step.to_string().len()));
    out
}

/// The one-line verdict a curve cannot show: where it started, where it ended,
/// and whether that is a descent.
///
/// Deliberately separate from [`render`]. A plot is a shape; this is a claim,
/// and the two should not be entangled in one function that a caller cannot
/// take half of.
pub fn summary(points: &[(usize, f32)]) -> String {
    let finite: Vec<(usize, f32)> = points.iter().copied().filter(|(_, l)| l.is_finite()).collect();
    if finite.is_empty() {
        return "loss: no finite values".to_string();
    }
    let (first_step, first) = finite[0];
    let (last_step, last) = finite[finite.len() - 1];
    let (best_step, best) = finite
        .iter()
        .copied()
        .fold((first_step, first), |acc, p| if p.1 < acc.1 { p } else { acc });

    let direction = if last < first {
        "descending"
    } else if last > first {
        "rising"
    } else {
        "flat"
    };
    format!(
        "loss {first:.6} (step {first_step}) -> {last:.6} (step {last_step}), \
         {direction}; best {best:.6} at step {best_step}"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn series(losses: &[f32]) -> Vec<(usize, f32)> {
        losses.iter().enumerate().map(|(i, &l)| (i, l)).collect()
    }

    #[test]
    fn an_empty_series_draws_nothing_rather_than_an_empty_box() {
        assert!(render(&[]).is_empty());
    }

    #[test]
    fn a_descending_run_draws_a_curve_with_its_real_range_on_the_axis() {
        let points = series(&[1.0, 0.8, 0.6, 0.4, 0.2]);
        let plot = render(&points);

        assert!(plot.contains("5 points"), "{plot}");
        assert!(plot.contains("steps 0..4"), "{plot}");
        // The axis labels must carry the real loss range, not a rounded guess.
        assert!(plot.contains("1.0000"), "top of the axis missing: {plot}");
        assert!(plot.contains('*'), "the curve must actually be drawn");
        assert_eq!(
            plot.lines().filter(|l| l.contains('|')).count(),
            HEIGHT,
            "the plot must be exactly HEIGHT rows tall"
        );
    }

    /// The minimum must land inside the plot, not fall off the bottom edge.
    #[test]
    fn the_lowest_point_is_drawn_on_the_last_row() {
        let plot = render(&series(&[1.0, 0.5, 0.0]));
        let rows: Vec<&str> = plot.lines().filter(|l| l.contains('|')).collect();
        assert!(
            rows[rows.len() - 1].contains('*'),
            "the minimum must be on the bottom row: {plot}"
        );
    }

    #[test]
    fn a_flat_run_is_drawn_rather_than_dividing_by_a_zero_span() {
        let plot = render(&series(&[0.5; 10]));
        assert!(plot.contains('*'), "a flat run still has a curve: {plot}");
        assert!(!plot.contains("NaN"), "a zero span must not produce NaN: {plot}");
    }

    #[test]
    fn a_single_point_draws_without_panicking() {
        let plot = render(&[(7, 0.25)]);
        assert!(plot.contains("1 points"));
        assert!(plot.contains("steps 7..7"));
        assert!(plot.contains('*'));
    }

    /// A run that produced only NaN is a real observation, and it is not a
    /// curve. Say so rather than drawing an empty plot.
    #[test]
    fn an_all_non_finite_series_says_so_instead_of_plotting() {
        let out = render(&series(&[f32::NAN, f32::INFINITY, f32::NAN]));
        assert!(out.contains("non-finite"), "{out}");
        assert!(!out.contains('*'), "nothing may be plotted: {out}");
    }

    #[test]
    fn non_finite_points_are_skipped_but_the_rest_still_plot() {
        let plot = render(&series(&[1.0, f32::NAN, 0.5, 0.1]));
        assert!(plot.contains('*'));
        assert!(!plot.contains("non-finite"), "some values were finite: {plot}");
        assert!(plot.contains("4 points"), "the skipped point is still counted");
    }

    /// More points than columns must compress, not overflow the width.
    #[test]
    fn a_long_run_compresses_into_the_fixed_width() {
        let losses: Vec<f32> = (0..1000).map(|i| 1.0 - i as f32 * 0.0009).collect();
        let plot = render(&series(&losses));
        for row in plot.lines().filter(|l| l.contains('|')) {
            let drawn = row.split('|').nth(1).unwrap_or("");
            assert!(
                drawn.len() <= WIDTH,
                "row is {} wide, max {WIDTH}: {row:?}",
                drawn.len()
            );
        }
        assert!(plot.contains("1000 points over 60 columns"), "{plot}");
    }

    /// D-19: a thinned series has uneven indices. Columns are spaced by
    /// position, so a thinned region must not read as a gap.
    #[test]
    fn a_sampled_series_with_uneven_indices_plots_without_gaps() {
        let points = vec![(0usize, 1.0f32), (16, 0.7), (32, 0.5), (48, 0.4), (49, 0.35)];
        let plot = render(&points);
        assert!(plot.contains("steps 0..49"), "{plot}");
        assert!(plot.contains("5 points"), "{plot}");
        // Every column holds a point, because columns == points here.
        let first_row = plot.lines().find(|l| l.contains('|')).unwrap();
        let drawn = first_row.split('|').nth(1).unwrap();
        assert_eq!(drawn.trim_end().len(), 1, "the max is one column wide: {drawn:?}");
    }

    // -----------------------------------------------------------------------
    // summary
    // -----------------------------------------------------------------------

    #[test]
    fn the_summary_names_the_direction_and_the_best_point() {
        let s = summary(&series(&[1.0, 0.5, 0.25]));
        assert!(s.contains("descending"), "{s}");
        assert!(s.contains("best 0.250000 at step 2"), "{s}");

        let s = summary(&series(&[0.25, 0.5, 1.0]));
        assert!(s.contains("rising"), "{s}");
        assert!(s.contains("best 0.250000 at step 0"), "{s}");

        let s = summary(&series(&[0.5, 0.5]));
        assert!(s.contains("flat"), "{s}");
    }

    /// The best point is the minimum, which is not always the last — the shape
    /// an overfitting run has.
    #[test]
    fn the_summary_distinguishes_the_best_point_from_the_final_one() {
        let s = summary(&series(&[1.0, 0.1, 0.9]));
        assert!(s.contains("best 0.100000 at step 1"), "{s}");
        assert!(s.contains("-> 0.900000 (step 2)"), "{s}");
        // 1.0 -> 0.9 is a descent by endpoints, even though the run gave back
        // most of what it gained. The direction describes the endpoints; the
        // best point is what says the run peaked in the middle.
        assert!(s.contains("descending"), "{s}");
    }

    #[test]
    fn the_summary_reports_no_finite_values_rather_than_inventing_one() {
        assert_eq!(summary(&[]), "loss: no finite values");
        assert_eq!(summary(&series(&[f32::NAN])), "loss: no finite values");
    }
}
