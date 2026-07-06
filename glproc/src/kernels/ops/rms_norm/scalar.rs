pub fn run(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    run_into(x, w, eps, &mut out);
    out
}

/// Allocation-free variant for the decode loop: writes into `out`.
/// `out.len()` must equal `x.len()`. `out` may alias-free overlap is not
/// supported — pass a distinct buffer.
pub fn run_into(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    debug_assert_eq!(out.len(), x.len());
    let mean_sq = x.iter().map(|&v| v * v).sum::<f32>() / x.len().max(1) as f32;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    for ((o, &xi), &wi) in out.iter_mut().zip(x).zip(w) {
        *o = xi * inv * wi;
    }
}
