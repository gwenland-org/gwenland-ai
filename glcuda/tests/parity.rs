//! Numerical parity tests: every glcuda kernel against the glproc scalar
//! ground truth, within the per-operation ε defined in ArchGLML_X2 §8
//! (Principle 2).
//!
//! On machines without a CUDA driver/device every test skips with a note —
//! the suite must be runnable (and green) everywhere, and *meaningful* on
//! CUDA hardware. On a GPU runner, execute with `--test-threads=1` so the
//! VRAM leak check is not perturbed by concurrent tests.

use glcuda::buffer::BackendBuffer;
use glcuda::driver::{cuda_available, Cuda};
use glcuda::kernels::{rope_tables, KernelSet};

// Per-operation tolerances from the architecture document (§8).
const EPS_MATMUL: f32 = 1e-5;
const EPS_RMSNORM: f32 = 1e-6;
const EPS_SOFTMAX: f32 = 1e-5;
const EPS_ROPE: f32 = 1e-7;
const EPS_SWIGLU: f32 = 1e-6;

/// Probe the GPU, or skip the test with an explicit note.
fn gpu() -> Option<(Cuda, KernelSet)> {
    if !cuda_available() {
        eprintln!("SKIP: no CUDA driver/device on this machine");
        return None;
    }
    let cuda = Cuda::probe().expect("driver reported available; probe must succeed");
    let kernels = KernelSet::load(&cuda).expect("PTX must JIT on sm_70+");
    Some((cuda, kernels))
}

/// Deterministic pseudo-random values in [-scale, scale] — same generator
/// family as glproc's runner tests.
fn randv(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            ((state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32 / (1u64 << 24) as f32
                - 0.5)
                * 2.0
                * scale
        })
        .collect()
}

fn assert_close(got: &[f32], want: &[f32], eps: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= eps,
            "{what}[{i}]: gpu {g} vs cpu {w} (|diff| {} > ε {eps})",
            (g - w).abs()
        );
    }
}

/// Upload a slice into a fresh region of `buf`.
fn upload(cuda: &Cuda, buf: &mut BackendBuffer, data: &[f32]) -> u64 {
    let s = buf.alloc_f32(data.len()).unwrap();
    cuda.htod_f32(s.dptr, data).unwrap();
    s.dptr
}

#[test]
fn gemv_matches_glproc_scalar() {
    let Some((cuda, k)) = gpu() else { return };
    // Second case has ragged dimensions (not multiples of the warp size)
    // to exercise the strided-loop exit guards.
    for (out_dim, in_dim, seed) in [(300usize, 896usize, 1u64), (37, 129, 2)] {
        let w = randv(out_dim * in_dim, seed, 0.1);
        let x = randv(in_dim, seed + 10, 1.0);
        let mut want = vec![0f32; out_dim];
        glproc::kernels::matmul::scalar::run_matvec(&w, &x, &mut want, out_dim, in_dim);

        let mut buf = BackendBuffer::new(&cuda, ((out_dim * in_dim + in_dim + out_dim) * 4 + 4096) as u64).unwrap();
        let dw = upload(&cuda, &mut buf, &w);
        let dx = upload(&cuda, &mut buf, &x);
        let dy = buf.alloc_f32(out_dim).unwrap().dptr;
        k.gemv(&cuda, dw, dx, dy, out_dim as u32, in_dim as u32).unwrap();
        cuda.synchronize().unwrap();
        let mut got = vec![0f32; out_dim];
        cuda.dtoh_f32(&mut got, dy).unwrap();
        buf.free(&cuda).unwrap();

        assert_close(&got, &want, EPS_MATMUL, "gemv");
    }
}

/// The quantized GEMV must equal "dequantize on the host, then f32 GEMV" —
/// the in-register dequant is an FP32 mul of exactly converted operands,
/// so only accumulation order separates the two.
#[test]
fn gemv_q8_0_matches_dequantized_reference() {
    let Some((cuda, k)) = gpu() else { return };
    let (out_dim, in_dim) = (48usize, 128usize); // whole Q8_0 blocks per row
    let w_f32 = randv(out_dim * in_dim, 40, 0.1);
    let blocks = glproc::kernels::dequant::q8_0::scalar::quantize(&w_f32);
    // CPU ground truth: dequantized blocks (not the original f32 — Q8_0 is
    // lossy) through the scalar matvec.
    let w_deq = glproc::kernels::dequant::q8_0::scalar::run(&blocks);
    let x = randv(in_dim, 41, 1.0);
    let mut want = vec![0f32; out_dim];
    glproc::kernels::matmul::scalar::run_matvec(&w_deq, &x, &mut want, out_dim, in_dim);

    let mut buf =
        BackendBuffer::new(&cuda, (blocks.len() + (in_dim + out_dim) * 4 + 4096) as u64).unwrap();
    let dw = buf.alloc(blocks.len() as u64).unwrap().dptr;
    cuda.htod(dw, &blocks).unwrap();
    let dx = upload(&cuda, &mut buf, &x);
    let dy = buf.alloc_f32(out_dim).unwrap().dptr;
    k.gemv_q8_0(&cuda, dw, dx, dy, out_dim as u32, in_dim as u32).unwrap();
    cuda.synchronize().unwrap();
    let mut got = vec![0f32; out_dim];
    cuda.dtoh_f32(&mut got, dy).unwrap();
    buf.free(&cuda).unwrap();

    assert_close(&got, &want, EPS_MATMUL, "gemv_q8_0");
}

#[test]
fn rms_norm_matches_glproc_scalar() {
    let Some((cuda, k)) = gpu() else { return };
    // 896 = Qwen2.5-0.5B dim; 8 exercises a block far wider than the data.
    for (dim, seed) in [(896usize, 3u64), (8, 4)] {
        let x = randv(dim, seed, 2.0);
        let w = randv(dim, seed + 10, 1.0);
        let eps = 1e-5f32;
        let mut want = vec![0f32; dim];
        glproc::kernels::ops::rms_norm::scalar::run_into(&x, &w, eps, &mut want);

        let mut buf = BackendBuffer::new(&cuda, (dim * 3 * 4 + 4096) as u64).unwrap();
        let dx = upload(&cuda, &mut buf, &x);
        let dw = upload(&cuda, &mut buf, &w);
        let dout = buf.alloc_f32(dim).unwrap().dptr;
        k.rms_norm(&cuda, dx, dw, dout, dim as u32, eps).unwrap();
        cuda.synchronize().unwrap();
        let mut got = vec![0f32; dim];
        cuda.dtoh_f32(&mut got, dout).unwrap();
        buf.free(&cuda).unwrap();

        assert_close(&got, &want, EPS_RMSNORM, "rms_norm");
    }
}

#[test]
fn silu_mul_matches_glproc_scalar() {
    let Some((cuda, k)) = gpu() else { return };
    let n = 4864usize; // Qwen2.5-0.5B hidden_dim
    let gate = randv(n, 5, 4.0);
    let up = randv(n, 6, 4.0);
    let mut want = gate.clone();
    glproc::kernels::ops::silu::scalar::run(&mut want, &up);

    let mut buf = BackendBuffer::new(&cuda, (n * 2 * 4 + 4096) as u64).unwrap();
    let dgate = upload(&cuda, &mut buf, &gate);
    let dup = upload(&cuda, &mut buf, &up);
    k.silu_mul(&cuda, dgate, dup, n as u32).unwrap();
    cuda.synchronize().unwrap();
    let mut got = vec![0f32; n];
    cuda.dtoh_f32(&mut got, dgate).unwrap();
    buf.free(&cuda).unwrap();

    assert_close(&got, &want, EPS_SWIGLU, "silu_mul");
}

#[test]
fn add_is_exact() {
    let Some((cuda, k)) = gpu() else { return };
    let n = 1000usize;
    let y = randv(n, 7, 1.0);
    let x = randv(n, 8, 1.0);
    let want: Vec<f32> = y.iter().zip(&x).map(|(a, b)| a + b).collect();

    let mut buf = BackendBuffer::new(&cuda, (n * 2 * 4 + 4096) as u64).unwrap();
    let dy = upload(&cuda, &mut buf, &y);
    let dx = upload(&cuda, &mut buf, &x);
    k.add(&cuda, dy, dx, n as u32).unwrap();
    cuda.synchronize().unwrap();
    let mut got = vec![0f32; n];
    cuda.dtoh_f32(&mut got, dy).unwrap();
    buf.free(&cuda).unwrap();

    // Single IEEE add, identical operands — must be bit-exact.
    assert_close(&got, &want, 0.0, "add");
}

/// Scalar softmax with exact `exp` — glproc's attention::softmax algorithm,
/// but pinned to the scalar ground-truth exponential (the dispatched
/// fast_exp path on an AVX2 test box carries its own ~1e-4 approximation,
/// which would swamp the 1e-5 contract this test enforces).
fn softmax_ref(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        let n = x.len() as f32;
        x.fill(1.0 / n);
        return;
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in x.iter_mut() {
            *v /= sum;
        }
    }
}

#[test]
fn softmax_matches_scalar_reference() {
    let Some((cuda, k)) = gpu() else { return };
    for (n, scale, seed) in [(1000usize, 0.125f32, 9u64), (3, 1.0, 10)] {
        let s = randv(n, seed, 8.0);
        let mut want: Vec<f32> = s.iter().map(|v| v * scale).collect();
        softmax_ref(&mut want);

        let mut buf = BackendBuffer::new(&cuda, (n * 4 + 4096) as u64).unwrap();
        let ds = upload(&cuda, &mut buf, &s);
        k.softmax_scale(&cuda, ds, n as u32, scale).unwrap();
        cuda.synchronize().unwrap();
        let mut got = vec![0f32; n];
        cuda.dtoh_f32(&mut got, ds).unwrap();
        buf.free(&cuda).unwrap();

        assert_close(&got, &want, EPS_SOFTMAX, "softmax");
        let total: f32 = got.iter().sum();
        assert!((total - 1.0).abs() < 1e-5, "softmax must sum to 1, got {total}");
    }
}

/// Mirror of glproc's `runner::rope` (private there), the CPU ground truth.
fn rope_ref(x: &mut [f32], pos: usize, n_heads: usize, head_dim: usize, base: f32, neox: bool) {
    let half = head_dim / 2;
    for h in 0..n_heads {
        let seg = &mut x[h * head_dim..(h + 1) * head_dim];
        for i in 0..half {
            let freq = 1.0 / base.powf(2.0 * i as f32 / head_dim as f32);
            let theta = pos as f32 * freq;
            let (sin, cos) = theta.sin_cos();
            let (a, b) = if neox { (i, i + half) } else { (2 * i, 2 * i + 1) };
            let x0 = seg[a];
            let x1 = seg[b];
            seg[a] = x0 * cos - x1 * sin;
            seg[b] = x0 * sin + x1 * cos;
        }
    }
}

#[test]
fn rope_matches_reference_both_styles() {
    let Some((cuda, k)) = gpu() else { return };
    let (n_heads, head_dim, pos, base) = (4usize, 64usize, 17usize, 10_000.0f32);
    for neox in [false, true] {
        let x = randv(n_heads * head_dim, 11 + neox as u64, 1.0);
        let mut want = x.clone();
        rope_ref(&mut want, pos, n_heads, head_dim, base, neox);

        let (cos, sin) = rope_tables(pos, head_dim, base);
        let mut buf =
            BackendBuffer::new(&cuda, ((x.len() + head_dim) * 4 + 4096) as u64).unwrap();
        let dx = upload(&cuda, &mut buf, &x);
        let dcos = upload(&cuda, &mut buf, &cos);
        let dsin = upload(&cuda, &mut buf, &sin);
        k.rope(&cuda, dx, dcos, dsin, n_heads as u32, head_dim as u32, neox).unwrap();
        cuda.synchronize().unwrap();
        let mut got = vec![0f32; x.len()];
        cuda.dtoh_f32(&mut got, dx).unwrap();
        buf.free(&cuda).unwrap();

        // Host-computed tables + identical mul/sub ordering: the contract
        // is the tight element-wise ε (should in fact be bit-exact).
        assert_close(&got, &want, EPS_ROPE, &format!("rope(neox={neox})"));
    }
}

/// Single-query decode attention with exact-exp softmax — glproc's
/// `attention_one` algorithm pinned to scalar ground truth.
fn attention_ref(q: &[f32], k_cache: &[f32], v_cache: &[f32], head_dim: usize) -> Vec<f32> {
    let cached_len = k_cache.len() / head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut scores: Vec<f32> = (0..cached_len)
        .map(|t| {
            let k_row = &k_cache[t * head_dim..(t + 1) * head_dim];
            glproc::kernels::matmul::scalar::dot_f32(q, k_row) * scale
        })
        .collect();
    softmax_ref(&mut scores);
    let mut out = vec![0f32; head_dim];
    for (t, &w) in scores.iter().enumerate() {
        for d in 0..head_dim {
            out[d] += w * v_cache[t * head_dim + d];
        }
    }
    out
}

/// Decode attention composed exactly as the forward pass will run it:
/// gemv (Q·K over cache rows) → softmax_scale → gemv_t (weighted V sum).
#[test]
fn attention_decode_composition_matches_reference() {
    let Some((cuda, k)) = gpu() else { return };
    let (head_dim, cached_len) = (64usize, 100usize);
    let q = randv(head_dim, 20, 1.0);
    let kc = randv(cached_len * head_dim, 21, 1.0);
    let vc = randv(cached_len * head_dim, 22, 1.0);
    let want = attention_ref(&q, &kc, &vc, head_dim);

    let bytes = ((q.len() + kc.len() + vc.len() + cached_len + head_dim) * 4 + 8192) as u64;
    let mut buf = BackendBuffer::new(&cuda, bytes).unwrap();
    let dq = upload(&cuda, &mut buf, &q);
    let dk = upload(&cuda, &mut buf, &kc);
    let dv = upload(&cuda, &mut buf, &vc);
    let dscores = buf.alloc_f32(cached_len).unwrap().dptr;
    let dout = buf.alloc_f32(head_dim).unwrap().dptr;

    let scale = 1.0 / (head_dim as f32).sqrt();
    k.gemv(&cuda, dk, dq, dscores, cached_len as u32, head_dim as u32).unwrap();
    k.softmax_scale(&cuda, dscores, cached_len as u32, scale).unwrap();
    k.gemv_t(&cuda, dv, dscores, dout, cached_len as u32, head_dim as u32).unwrap();
    cuda.synchronize().unwrap();

    let mut got = vec![0f32; head_dim];
    cuda.dtoh_f32(&mut got, dout).unwrap();
    buf.free(&cuda).unwrap();

    assert_close(&got, &want, EPS_MATMUL, "attention_decode");
}

/// ADR-005 / M2 definition of done: the backend buffer must give VRAM back
/// exactly — free VRAM identical before and after a full alloc/use/free
/// cycle. Requires `--test-threads=1` to be meaningful (see module doc).
#[test]
fn backend_buffer_returns_vram_exactly() {
    let Some((cuda, k)) = gpu() else { return };
    let (free_before, _) = cuda.mem_get_info().unwrap();

    let mut buf = BackendBuffer::new(&cuda, 1 << 22).unwrap();
    let x = randv(1024, 30, 1.0);
    let d = upload(&cuda, &mut buf, &x);
    k.add(&cuda, d, d, 1024).unwrap(); // x + x — exercise a launch too
    cuda.synchronize().unwrap();
    let mut got = vec![0f32; 1024];
    cuda.dtoh_f32(&mut got, d).unwrap();
    buf.free(&cuda).unwrap();

    let (free_after, _) = cuda.mem_get_info().unwrap();
    assert_eq!(free_before, free_after, "backend buffer leaked VRAM");
    for (g, w) in got.iter().zip(&x) {
        assert_eq!(*g, w + w);
    }
}
