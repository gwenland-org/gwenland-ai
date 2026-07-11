//! Scaling analysis over a set of related sessions (batch / context / hardware
//! sweeps).
//!
//! Given several sessions that vary one axis (e.g. prompt length), this reports
//! how throughput scales along it — linear, sub-linear, or saturating. It reads
//! multiple sessions rather than one, so it lives here as a standalone helper
//! the CLI can call over an archive directory; the single-session pipeline in
//! [`super::summary`] does not invoke it.

/// One point on a scaling curve: an axis value and the throughput observed there.
#[derive(Debug, Clone, Copy)]
pub struct ScalePoint {
    /// The independent-axis value (batch size, context length, core count, ...).
    pub axis: f64,
    /// Observed throughput at that axis value (tokens/second).
    pub throughput: f64,
}

/// A qualitative scaling verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scaling {
    /// Throughput grows roughly proportionally with the axis.
    Linear,
    /// Throughput grows, but less than proportionally.
    SubLinear,
    /// Throughput has flattened — adding more of the axis stops helping.
    Saturating,
    /// Too few points to say.
    Insufficient,
}

/// Classify how `points` scale. Expects points sorted ascending by axis; needs
/// at least two to say anything.
pub fn classify(points: &[ScalePoint]) -> Scaling {
    if points.len() < 2 {
        return Scaling::Insufficient;
    }
    let first = points[0];
    let last = points[points.len() - 1];
    if first.axis <= 0.0 || first.throughput <= 0.0 {
        return Scaling::Insufficient;
    }
    let axis_ratio = last.axis / first.axis;
    let tput_ratio = last.throughput / first.throughput;
    if axis_ratio <= 1.0 {
        return Scaling::Insufficient;
    }
    // Efficiency of scaling: how much of the axis growth turned into throughput.
    let scaling_eff = (tput_ratio - 1.0) / (axis_ratio - 1.0);
    if scaling_eff >= 0.85 {
        Scaling::Linear
    } else if scaling_eff >= 0.15 {
        Scaling::SubLinear
    } else {
        Scaling::Saturating
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pt(axis: f64, throughput: f64) -> ScalePoint {
        ScalePoint { axis, throughput }
    }

    #[test]
    fn doubling_throughput_on_doubling_axis_is_linear() {
        assert_eq!(classify(&[pt(1.0, 100.0), pt(2.0, 200.0)]), Scaling::Linear);
    }

    #[test]
    fn flat_throughput_saturates() {
        assert_eq!(classify(&[pt(1.0, 100.0), pt(4.0, 105.0)]), Scaling::Saturating);
    }
}
