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
// Per-operation tolerances. The doc (§8) lists aspirational values assuming
// identical rounding between engines; on real hardware the GPU contracts
// mul+add into FMA and uses approx transcendentals, so the achievable bound
// is ~1 ULP looser for the ops that do arithmetic on the payload. These are
// combined absolute-or-relative (see `assert_close`).
const EPS_MATMUL: f32 = 1e-5;
// Q8_0 GEMV quantizes BOTH operands to int8. The host reference dequantizes
// with div+round-half-away while the device uses rcp.rn+round-to-even, so a
// rare single-quantum flip can leave a ~1e-3 residual on the accumulated dot.
// 1e-3 is the honest bound for a fully-quantized matvec (a real kernel bug is
// orders of magnitude larger); f32/Q4 GEMVs keep the tight EPS_MATMUL.
const EPS_Q8_GEMV: f32 = 1e-3;
// Q4_K GEMV: both operands quantized (4-bit weights + int8 activations) AND
// the sub-block scales/mins are pre-multiplied into f16 at repack (~2^-12
// relative each). Q4_K is the lossiest format in the suite; 1e-2 is the
// architecture's stated tolerance for it (a real kernel/layout bug shows up
// orders of magnitude larger).
const EPS_Q4K_GEMV: f32 = 1e-2;
const EPS_RMSNORM: f32 = 1e-6;
const EPS_SOFTMAX: f32 = 1e-5;
// RoPE: doc says 1e-7 ("element-wise, no reduction"), but the device fuses
// x0*cos - x1*sin into an FMA the CPU computes as two rounded ops — a 1-ULP
// gap the 1e-7 bound cannot hold. 1e-6 is the honest element-wise tolerance.
const EPS_ROPE: f32 = 1e-6;
// SwiGLU: sigmoid via ex2.approx is a low-ULP approximation; 1e-5 absolute
// (or relative) covers it at the magnitudes real activations reach.
const EPS_SWIGLU: f32 = 1e-5;

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

/// Deterministic pseudo-random bytes.
fn randv_bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            (state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u8
        })
        .collect()
}

/// Mirror glcuda's `quantize_q8` (per-32-block amax/127 scale, round-to-
/// nearest, clamp) then dequantize. The Q8_0 GEMV quantizes its activation to
/// int8 on-device, so a faithful reference must dot the DEQUANTIZED activation
/// — not the full-precision one — or it disagrees by the ~1/127 activation
/// quantization error (which is what tripped both q8 GEMV tests otherwise).
fn q8_round_trip(x: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    for (blk_in, blk_out) in x.chunks(32).zip(out.chunks_mut(32)) {
        let amax = blk_in.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let scale = amax / 127.0;
        for (o, &v) in blk_out.iter_mut().zip(blk_in) {
            let q = if scale != 0.0 { (v / scale).round().clamp(-128.0, 127.0) } else { 0.0 };
            *o = q * scale;
        }
    }
    out
}

/// Mixed absolute-or-relative closeness. `eps` is the per-operation
/// tolerance from ArchGLML_X2 §8; it is honored as an *absolute* bound near
/// zero and as a *relative* bound for larger magnitudes. A fixed absolute
/// ε is the wrong model for float error — the GPU's FMA-contracted and
/// approx-transcendental results differ from the CPU's by ~1 ULP, whose
/// absolute size grows with the value (this is what tripped SwiGLU at ~10
/// and RoPE at ~1 on the first T4 run). `|g - w| <= eps * max(1, |w|)`
/// captures both regimes.
fn assert_close(got: &[f32], want: &[f32], eps: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let tol = eps * w.abs().max(1.0);
        assert!(
            (g - w).abs() <= tol,
            "{what}[{i}]: gpu {g} vs cpu {w} (|diff| {} > tol {tol} = eps {eps} * max(1,|w|))",
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
    // The GEMV quantizes x to int8 on-device; feed the reference the same.
    let x_dq = q8_round_trip(&x);
    let mut want = vec![0f32; out_dim];
    glproc::kernels::matmul::scalar::run_matvec(&w_deq, &x_dq, &mut want, out_dim, in_dim);

    let mut padded = Vec::with_capacity((blocks.len() / 34) * 36);
    for block in blocks.chunks_exact(34) {
        padded.extend_from_slice(&block[0..2]);
        padded.extend_from_slice(&[0, 0]);
        padded.extend_from_slice(&block[2..34]);
    }

    let mut buf =
        BackendBuffer::new(&cuda, (padded.len() + (in_dim + out_dim) * 4 + 4096) as u64).unwrap();
    let dw = buf.alloc(padded.len() as u64).unwrap().dptr;
    cuda.htod(dw, &padded).unwrap();
    let dx = upload(&cuda, &mut buf, &x);
    let d_qs = buf.alloc(in_dim as u64).unwrap().dptr;
    let d_scales = buf.alloc_f32(in_dim / 32).unwrap().dptr;
    k.quantize_q8(&cuda, dx, d_qs, d_scales, in_dim as u32).unwrap();
    let dy = buf.alloc_f32(out_dim).unwrap().dptr;
    k.gemv_q8_0(&cuda, dw, d_qs, d_scales, dy, out_dim as u32, in_dim as u32).unwrap();
    cuda.synchronize().unwrap();
    let mut got = vec![0f32; out_dim];
    cuda.dtoh_f32(&mut got, dy).unwrap();
    buf.free(&cuda).unwrap();

    assert_close(&got, &want, EPS_Q8_GEMV, "gemv_q8_0");
}

/// The SoA Q8_0 GEMV (contiguous qs + separate f16 scales) must match the same
/// dequantized reference as the AoS kernel — same math, different layout.
#[test]
fn gemv_q8_0_soa_matches_dequantized_reference() {
    let Some((cuda, k)) = gpu() else { return };
    let (out_dim, in_dim) = (48usize, 128usize);
    let w_f32 = randv(out_dim * in_dim, 40, 0.1);
    let blocks = glproc::kernels::dequant::q8_0::scalar::quantize(&w_f32);
    let w_deq = glproc::kernels::dequant::q8_0::scalar::run(&blocks);
    let x = randv(in_dim, 41, 1.0);
    let x_dq = q8_round_trip(&x);
    let mut want = vec![0f32; out_dim];
    glproc::kernels::matmul::scalar::run_matvec(&w_deq, &x_dq, &mut want, out_dim, in_dim);

    // Split the 34-byte blocks into contiguous qs + f16 scales (as the loader does).
    let n_blocks = blocks.len() / 34;
    let mut qs = Vec::with_capacity(n_blocks * 32);
    let mut scales = Vec::with_capacity(n_blocks * 2);
    for block in blocks.chunks_exact(34) {
        scales.extend_from_slice(&block[0..2]);
        qs.extend_from_slice(&block[2..34]);
    }

    let mut buf = BackendBuffer::new(
        &cuda,
        (qs.len() + scales.len() + (in_dim + out_dim) * 4 + 4096) as u64,
    )
    .unwrap();
    let dwqs = buf.alloc(qs.len() as u64).unwrap().dptr;
    cuda.htod(dwqs, &qs).unwrap();
    let dwsc = buf.alloc(scales.len() as u64).unwrap().dptr;
    cuda.htod(dwsc, &scales).unwrap();
    let dx = upload(&cuda, &mut buf, &x);
    let d_qs = buf.alloc(in_dim as u64).unwrap().dptr;
    let d_scales = buf.alloc_f32(in_dim / 32).unwrap().dptr;
    k.quantize_q8(&cuda, dx, d_qs, d_scales, in_dim as u32).unwrap();
    let dy = buf.alloc_f32(out_dim).unwrap().dptr;
    k.gemv_q8_0_soa(&cuda, dwqs, dwsc, d_qs, d_scales, dy, out_dim as u32, in_dim as u32)
        .unwrap();
    cuda.synchronize().unwrap();
    let mut got = vec![0f32; out_dim];
    cuda.dtoh_f32(&mut got, dy).unwrap();
    buf.free(&cuda).unwrap();

    assert_close(&got, &want, EPS_Q8_GEMV, "gemv_q8_0_soa");
}

/// M2.1 Task A: the native Q4_K SoA GEMV against the glproc scalar ground
/// truth. Weights are synthetic Q4_K super-blocks (random nibbles + packed
/// 6-bit scales, sane f16 d/dmin), dequantized by glproc for the reference;
/// the device gets the `repack::q4_k_to_soa` streams and the activation
/// int8-quantized on device, exactly as the forward pass runs it.
#[test]
fn gemv_q4_k_soa_matches_dequantized_reference() {
    let Some((cuda, k)) = gpu() else { return };
    let (out_dim, in_dim) = (48usize, 512usize); // 2 super-blocks per row
    let blocks_len = (in_dim / 256 * 144) * out_dim;
    let mut blocks = randv_bytes(blocks_len, 45);
    for block in blocks.chunks_exact_mut(144) {
        block[0..2].copy_from_slice(&0x1e66u16.to_le_bytes()); // d ~ 0.0016
        block[2..4].copy_from_slice(&0x1a66u16.to_le_bytes()); // dmin ~ 0.0008
    }
    // CPU ground truth: glproc's Q4_K dequant through the scalar matvec,
    // against the round-tripped (int8-quantized) activation the kernel sees.
    let w_deq = glproc::kernels::dequant::q4_k::scalar::run(&blocks).unwrap();
    let x = randv(in_dim, 46, 1.0);
    let x_dq = q8_round_trip(&x);
    let mut want = vec![0f32; out_dim];
    glproc::kernels::matmul::scalar::run_matvec(&w_deq, &x_dq, &mut want, out_dim, in_dim);

    let (wqs, wsc, wmn) = glcuda::repack::q4_k_to_soa(&blocks).unwrap();

    let mut buf = BackendBuffer::new(
        &cuda,
        (wqs.len() + wsc.len() + wmn.len() + (in_dim + out_dim) * 4 + 8192) as u64,
    )
    .unwrap();
    let dwqs = buf.alloc(wqs.len() as u64).unwrap().dptr;
    cuda.htod(dwqs, &wqs).unwrap();
    let dwsc = buf.alloc(wsc.len() as u64).unwrap().dptr;
    cuda.htod(dwsc, &wsc).unwrap();
    let dwmn = buf.alloc(wmn.len() as u64).unwrap().dptr;
    cuda.htod(dwmn, &wmn).unwrap();
    let dx = upload(&cuda, &mut buf, &x);
    let d_qs = buf.alloc(in_dim as u64).unwrap().dptr;
    let d_scales = buf.alloc_f32(in_dim / 32).unwrap().dptr;
    k.quantize_q8(&cuda, dx, d_qs, d_scales, in_dim as u32).unwrap();
    let dy = buf.alloc_f32(out_dim).unwrap().dptr;
    k.gemv_q4_k_soa(&cuda, dwqs, dwsc, dwmn, d_qs, d_scales, dy, out_dim as u32, in_dim as u32)
        .unwrap();
    cuda.synchronize().unwrap();
    let mut got = vec![0f32; out_dim];
    cuda.dtoh_f32(&mut got, dy).unwrap();
    buf.free(&cuda).unwrap();

    assert_close(&got, &want, EPS_Q4K_GEMV, "gemv_q4_k_soa");
}

#[test]
fn gemv_q4_0_matches_dequantized_reference() {
    let Some((cuda, k)) = gpu() else { return };
    let (out_dim, in_dim) = (48usize, 128usize); // whole Q4_0 blocks per row
    let blocks_len = (in_dim / 32 * 18) * out_dim;
    let mut blocks = randv_bytes(blocks_len, 40);
    // ensure scale (d) is not NaN/Inf for exact comparison
    for block in blocks.chunks_exact_mut(18) {
        block[0..2].copy_from_slice(&0x2e66u16.to_le_bytes()); // ~0.1
    }
    let w_deq = glproc::kernels::dequant::q4_0::scalar::run(&blocks);
    let x = randv(in_dim, 41, 1.0);
    let mut want = vec![0f32; out_dim];
    glproc::kernels::matmul::scalar::run_matvec(&w_deq, &x, &mut want, out_dim, in_dim);

    let mut buf =
        BackendBuffer::new(&cuda, (blocks.len() + (in_dim + out_dim) * 4 + 4096) as u64).unwrap();
    let dw = buf.alloc(blocks.len() as u64).unwrap().dptr;
    cuda.htod(dw, &blocks).unwrap();
    let dx = upload(&cuda, &mut buf, &x);
    let dy = buf.alloc_f32(out_dim).unwrap().dptr;
    k.gemv_q4_0(&cuda, dw, dx, dy, out_dim as u32, in_dim as u32).unwrap();
    cuda.synchronize().unwrap();
    let mut got = vec![0f32; out_dim];
    cuda.dtoh_f32(&mut got, dy).unwrap();
    buf.free(&cuda).unwrap();

    assert_close(&got, &want, EPS_MATMUL, "gemv_q4_0");
}

/// M2.1 Task B: the tensor-core batched GEMM against the same dequantized
/// reference as the dp4a GEMM. Runs only on sm_75+ (the module is not even
/// loaded below that). Ragged token count (5) exercises the padded-row read
/// / guarded-write contract; out=16 is two 8-row warp tiles.
#[test]
fn gemm_mma_q8_matches_dequantized_reference() {
    let Some((cuda, k)) = gpu() else { return };
    if !k.has_mma() {
        eprintln!("SKIP: device below sm_75 — no tensor-core module");
        return;
    }
    let (out_dim, in_dim, ntok) = (16usize, 64usize, 5usize);
    let ntok_pad = ntok.div_ceil(8) * 8;

    let w_f32 = randv(out_dim * in_dim, 50, 0.1);
    let blocks = glproc::kernels::dequant::q8_0::scalar::quantize(&w_f32);
    let w_deq = glproc::kernels::dequant::q8_0::scalar::run(&blocks);
    let x = randv(ntok_pad * in_dim, 51, 1.0);
    let x_dq = q8_round_trip(&x);
    let mut want = vec![0f32; ntok * out_dim];
    for t in 0..ntok {
        glproc::kernels::matmul::scalar::run_matvec(
            &w_deq,
            &x_dq[t * in_dim..(t + 1) * in_dim],
            &mut want[t * out_dim..(t + 1) * out_dim],
            out_dim,
            in_dim,
        );
    }

    let n_blocks = blocks.len() / 34;
    let mut qs = Vec::with_capacity(n_blocks * 32);
    let mut scales = Vec::with_capacity(n_blocks * 2);
    for block in blocks.chunks_exact(34) {
        scales.extend_from_slice(&block[0..2]);
        qs.extend_from_slice(&block[2..34]);
    }

    let bytes = (qs.len() + scales.len() + (ntok_pad * in_dim) * 5 + ntok * out_dim * 4 + 8192) as u64;
    let mut buf = BackendBuffer::new(&cuda, bytes).unwrap();
    let dwqs = buf.alloc(qs.len() as u64).unwrap().dptr;
    cuda.htod(dwqs, &qs).unwrap();
    let dwsc = buf.alloc(scales.len() as u64).unwrap().dptr;
    cuda.htod(dwsc, &scales).unwrap();
    let dx = upload(&cuda, &mut buf, &x);
    // Quantize all padded rows in one pass, exactly as prefill does.
    let d_qs = buf.alloc((ntok_pad * in_dim) as u64).unwrap().dptr;
    let d_scales = buf.alloc_f32(ntok_pad * in_dim / 32).unwrap().dptr;
    k.quantize_q8(&cuda, dx, d_qs, d_scales, (ntok_pad * in_dim) as u32).unwrap();
    let dy = buf.alloc_f32(ntok * out_dim).unwrap().dptr;
    k.gemm_mma_q8(&cuda, dwqs, dwsc, d_qs, d_scales, dy, out_dim as u32, in_dim as u32, ntok as u32)
        .unwrap();
    cuda.synchronize().unwrap();
    let mut got = vec![0f32; ntok * out_dim];
    cuda.dtoh_f32(&mut got, dy).unwrap();
    buf.free(&cuda).unwrap();

    assert_close(&got, &want, EPS_Q8_GEMV, "gemm_mma_q8");
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

        // The kernel indexes cos/sin at row `pos` (read from device). Build
        // the table for exactly this `pos` as row 0 and pass a device pos=0,
        // so row 0 holds this position's cos/sin.
        let (cos, sin) = rope_tables(pos, head_dim, base);
        let mut buf =
            BackendBuffer::new(&cuda, ((x.len() + head_dim) * 4 + 4096) as u64).unwrap();
        let dx = upload(&cuda, &mut buf, &x);
        let dcos = upload(&cuda, &mut buf, &cos);
        let dsin = upload(&cuda, &mut buf, &sin);
        let dpos = buf.alloc(4).unwrap().dptr;
        cuda.htod(dpos, &0u32.to_ne_bytes()).unwrap();
        k.rope(&cuda, dx, dcos, dsin, n_heads as u32, head_dim as u32, neox, dpos).unwrap();
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

/// The fused all-heads decode-attention kernel (`gl_attn_decode_f32`, M2.1)
/// must match the per-head reference for every head, including GQA where
/// several query heads share one KV head. K/V are laid out exactly as the
/// device KV cache does: `[kv_head][seq][dim]` with a `head_stride` of
/// `max_context * head_dim` elements between KV heads.
#[test]
fn fused_attn_decode_matches_per_head_reference() {
    let Some((cuda, k)) = gpu() else { return };
    let (head_dim, n_heads, n_kv, cached_len, max_ctx) = (64usize, 8usize, 2usize, 100usize, 128usize);
    let heads_per_kv = n_heads / n_kv;
    let head_stride = max_ctx * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Queries for all heads.
    let q = randv(n_heads * head_dim, 50, 1.0);
    // K and V regions sized to the full cache stride per KV head (only the
    // first `cached_len` rows of each are populated / read).
    let mut kc = vec![0f32; n_kv * head_stride];
    let mut vc = vec![0f32; n_kv * head_stride];
    for kvh in 0..n_kv {
        let src_k = randv(cached_len * head_dim, 60 + kvh as u64, 1.0);
        let src_v = randv(cached_len * head_dim, 70 + kvh as u64, 1.0);
        kc[kvh * head_stride..kvh * head_stride + src_k.len()].copy_from_slice(&src_k);
        vc[kvh * head_stride..kvh * head_stride + src_v.len()].copy_from_slice(&src_v);
    }

    // Reference: run the per-head attention for each query head against its
    // own KV head's first `cached_len` rows.
    let mut want = vec![0f32; n_heads * head_dim];
    for h in 0..n_heads {
        let kvh = h / heads_per_kv;
        let krows = &kc[kvh * head_stride..kvh * head_stride + cached_len * head_dim];
        let vrows = &vc[kvh * head_stride..kvh * head_stride + cached_len * head_dim];
        let out = attention_ref(&q[h * head_dim..(h + 1) * head_dim], krows, vrows, head_dim);
        want[h * head_dim..(h + 1) * head_dim].copy_from_slice(&out);
    }

    let bytes = ((q.len() + kc.len() + vc.len() + n_heads * head_dim) * 4 + 8192) as u64;
    let mut buf = BackendBuffer::new(&cuda, bytes).unwrap();
    let dq = upload(&cuda, &mut buf, &q);
    let dk = upload(&cuda, &mut buf, &kc);
    let dv = upload(&cuda, &mut buf, &vc);
    let dout = buf.alloc_f32(n_heads * head_dim).unwrap().dptr;
    // cached_len is now read from device memory (token-invariant graph args).
    let dclen = buf.alloc(4).unwrap().dptr;
    cuda.htod(dclen, &(cached_len as u32).to_ne_bytes()).unwrap();

    k.attn_decode(
        &cuda,
        dq,
        dk,
        dv,
        dout,
        n_heads as u32,
        head_dim as u32,
        dclen,
        heads_per_kv as u32,
        head_stride as u32,
        scale,
    )
    .unwrap();
    cuda.synchronize().unwrap();

    let mut got = vec![0f32; n_heads * head_dim];
    cuda.dtoh_f32(&mut got, dout).unwrap();
    buf.free(&cuda).unwrap();

    assert_close(&got, &want, EPS_MATMUL, "fused_attn_decode");
}

/// M2.2: gl_kv_write must place each KV head's row at the device-`pos`
/// slot of its cache region — the graph-static replacement for the per-head
/// cuMemcpyDtoD. Write two heads' rows at pos=3, read them back from the
/// computed offsets.
#[test]
fn kv_write_places_rows_at_device_pos() {
    let Some((cuda, k)) = gpu() else { return };
    let (head_dim, n_kv, max_ctx, pos) = (64usize, 2usize, 128usize, 3usize);
    let head_stride = max_ctx * head_dim;
    let src = randv(n_kv * head_dim, 90, 1.0); // both heads' rows, contiguous

    let mut buf = BackendBuffer::new(&cuda, ((n_kv * head_stride + src.len()) * 4 + 4096) as u64).unwrap();
    let dst = buf.alloc_f32(n_kv * head_stride).unwrap().dptr; // zeroed region
    cuda.htod_f32(dst, &vec![0f32; n_kv * head_stride]).unwrap();
    let dsrc = upload(&cuda, &mut buf, &src);
    let dpos = buf.alloc(4).unwrap().dptr;
    cuda.htod(dpos, &(pos as u32).to_ne_bytes()).unwrap();

    k.kv_write(&cuda, dst, dsrc, dpos, head_dim as u32, n_kv as u32, head_stride as u32).unwrap();
    cuda.synchronize().unwrap();

    let mut got = vec![0f32; n_kv * head_stride];
    cuda.dtoh_f32(&mut got, dst).unwrap();
    buf.free(&cuda).unwrap();

    // Each head h's row must land at [h*head_stride + pos*head_dim ..][..head_dim].
    for h in 0..n_kv {
        let off = h * head_stride + pos * head_dim;
        for d in 0..head_dim {
            assert_eq!(
                got[off + d],
                src[h * head_dim + d],
                "kv_write head {h} elem {d} landed wrong"
            );
        }
    }
}

/// M2.2 stage 1: prove CUDA graph capture + replay works on this hardware
/// before wiring it into the runner. Capture a fixed sequence of kernel
/// launches (three `gl_add`s) into a graph, then replay the graph N times
/// and confirm the arithmetic matches doing the launches directly — i.e.
/// the captured graph really executes the recorded work each replay.
#[test]
fn cuda_graph_capture_replay_executes() {
    let Some((cuda, k)) = gpu() else { return };
    let n = 1024usize;
    let base = randv(n, 80, 1.0);
    let addend = vec![1.0f32; n];

    let mut buf = BackendBuffer::new(&cuda, (n * 2 * 4 + 4096) as u64).unwrap();
    let acc = buf.alloc_f32(n).unwrap().dptr;
    let one = upload(&cuda, &mut buf, &addend);
    cuda.htod_f32(acc, &base).unwrap();

    // Capture "acc += one" three times into one graph. During capture the
    // launches are recorded, not executed, so acc is unchanged afterward.
    let graph = cuda
        .capture(|| {
            k.add(&cuda, acc, one, n as u32)?;
            k.add(&cuda, acc, one, n as u32)?;
            k.add(&cuda, acc, one, n as u32)?;
            Ok(())
        })
        .unwrap();

    // Two replays => acc should be base + 3 + 3 = base + 6.
    cuda.graph_launch(&graph).unwrap();
    cuda.graph_launch(&graph).unwrap();

    let mut got = vec![0f32; n];
    cuda.dtoh_f32(&mut got, acc).unwrap();
    buf.free(&cuda).unwrap();

    for (g, b) in got.iter().zip(&base) {
        assert_eq!(*g, b + 6.0, "graph replay did not execute the recorded adds");
    }
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
