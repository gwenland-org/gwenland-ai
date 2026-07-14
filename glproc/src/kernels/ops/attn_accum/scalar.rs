//! Scalar ground truth for attention value accumulation.

/// `out[d] = Σ_t weights[t] * v_cache[t * head_dim + d]`.
///
/// `v_cache` holds `weights.len()` contiguous rows of `head_dim` floats.
/// `out` is overwritten, not accumulated.
pub fn run(weights: &[f32], v_cache: &[f32], out: &mut [f32], head_dim: usize) {
    debug_assert_eq!(out.len(), head_dim);
    debug_assert!(v_cache.len() >= weights.len() * head_dim);

    out.fill(0.0);
    for (t, &w) in weights.iter().enumerate() {
        let v_row = &v_cache[t * head_dim..(t + 1) * head_dim];
        for (o, &v) in out.iter_mut().zip(v_row) {
            *o += w * v;
        }
    }
}
