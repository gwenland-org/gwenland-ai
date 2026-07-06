/// Dot product: `sum(a[i] * b[i])`. In-place, no allocation.
/// Ground truth for the SIMD `dot_f32` kernels.
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn run(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);
    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        let c_row = &mut c[i * n..(i + 1) * n];
        c_row.fill(0.0);
        for (p, &a_ip) in a_row.iter().enumerate() {
            let b_row = &b[p * n..(p + 1) * n];
            for (j, &b_pj) in b_row.iter().enumerate() {
                c_row[j] += a_ip * b_pj;
            }
        }
    }
}

pub fn run_t(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), n * k);
    debug_assert_eq!(c.len(), m * n);
    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        for j in 0..n {
            let b_row = &b[j * k..(j + 1) * k];
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a_row[p] * b_row[p];
            }
            c[i * n + j] = acc;
        }
    }
}

pub fn run_matvec(w: &[f32], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    debug_assert_eq!(w.len(), out_dim * in_dim);
    debug_assert_eq!(x.len(), in_dim);
    debug_assert_eq!(y.len(), out_dim);
    for o in 0..out_dim {
        let row = &w[o * in_dim..(o + 1) * in_dim];
        let mut acc = 0.0f32;
        for p in 0..in_dim {
            acc += row[p] * x[p];
        }
        y[o] = acc;
    }
}
