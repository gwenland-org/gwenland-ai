use std::arch::x86_64::*;

/// AVX-512F dot product: 16 f32 per register, two accumulators to hide FMA
/// latency, scalar tail for `len % 16 != 0`.
///
/// # Safety
/// Caller must ensure the CPU supports AVX-512F.
#[target_feature(enable = "avx512f", enable = "avx512bw")]
pub unsafe fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let pa = a.as_ptr();
    let pb = b.as_ptr();

    // Two independent accumulators: FMA has ~4-cycle latency, so a single
    // accumulator chain would stall; two chains keep the port busy.
    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();
    let mut i = 0;
    while i + 32 <= n {
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(pa.add(i)), _mm512_loadu_ps(pb.add(i)), acc0);
        acc1 = _mm512_fmadd_ps(
            _mm512_loadu_ps(pa.add(i + 16)),
            _mm512_loadu_ps(pb.add(i + 16)),
            acc1,
        );
        i += 32;
    }
    while i + 16 <= n {
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(pa.add(i)), _mm512_loadu_ps(pb.add(i)), acc0);
        i += 16;
    }

    let mut tmp = [0.0f32; 16];
    _mm512_storeu_ps(tmp.as_mut_ptr(), _mm512_add_ps(acc0, acc1));
    let mut sum = tmp.iter().sum::<f32>();
    while i < n {
        sum += *pa.add(i) * *pb.add(i);
        i += 1;
    }
    sum
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
pub unsafe fn run(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        let c_row = &mut c[i * n..(i + 1) * n];
        c_row.fill(0.0);
        for p in 0..k {
            let a_val = _mm512_set1_ps(a_row[p]);
            let b_row = b[p * n..(p + 1) * n].as_ptr();
            let mut j = 0;
            while j + 16 <= n {
                let b_vec = _mm512_loadu_ps(b_row.add(j) as *const _);
                let c_vec = _mm512_loadu_ps(c_row.as_mut_ptr().add(j) as *const _);
                let res = _mm512_fmadd_ps(a_val, b_vec, c_vec);
                _mm512_storeu_ps(c_row.as_mut_ptr().add(j) as *mut _, res);
                j += 16;
            }
            while j < n {
                c_row[j] += a_row[p] * *b_row.add(j);
                j += 1;
            }
        }
    }
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
pub unsafe fn run_t(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        run_matvec(b, &a[i * k .. (i + 1) * k], &mut c[i * n .. (i + 1) * n], n, k);
    }
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
pub unsafe fn run_matvec(w: &[f32], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    for o in 0..out_dim {
        let mut acc = _mm512_setzero_ps();
        let row = w[o * in_dim .. (o + 1) * in_dim].as_ptr();
        let x_ptr = x.as_ptr();
        
        let mut p = 0;
        while p + 16 <= in_dim {
            let r_vec = _mm512_loadu_ps(row.add(p) as *const _);
            let x_vec = _mm512_loadu_ps(x_ptr.add(p) as *const _);
            acc = _mm512_fmadd_ps(r_vec, x_vec, acc);
            p += 16;
        }
        
        let mut tmp = [0.0f32; 16];
        _mm512_storeu_ps(tmp.as_mut_ptr() as *mut _, acc);
        let mut sum = tmp.iter().sum::<f32>();
        
        while p < in_dim {
            sum += *row.add(p) * *x_ptr.add(p);
            p += 1;
        }
        
        y[o] = sum;
    }
}
