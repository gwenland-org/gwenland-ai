//! Scalar ground truth for the in-place softmax.
//!
//! This is the body that used to live in `attention::softmax`, moved here
//! verbatim so the SIMD path has an unchanged reference to be validated
//! against.

/// Numerically stable in-place softmax.
pub fn run(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        // All -inf (fully masked row) — degenerate; spread uniformly.
        let n = x.len() as f32;
        x.fill(1.0 / n);
        return;
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        // X5 §4.8: fast_exp (range-reduced polynomial, ~1e-4 rel err) in
        // place of f32::exp — well under the weights' quantization noise.
        *v = crate::kernels::fast_exp(*v - max);
        sum += *v;
    }
    if sum > 0.0 {
        for v in x.iter_mut() {
            *v /= sum;
        }
    }
}
