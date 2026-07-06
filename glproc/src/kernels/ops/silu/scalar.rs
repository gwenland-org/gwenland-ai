//! Scalar ground truth for the fused SwiGLU gating.

/// `gate[i] = silu(gate[i]) * up[i]`, in place.
pub fn run(gate: &mut [f32], up: &[f32]) {
    debug_assert_eq!(gate.len(), up.len());
    for (g, &u) in gate.iter_mut().zip(up) {
        let x = *g;
        *g = x / (1.0 + crate::kernels::ops::fast_exp::scalar::run(-x)) * u;
    }
}
