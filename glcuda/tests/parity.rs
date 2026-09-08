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
use glcuda::repack::q8_0_soa_to_bstage;

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
            ((state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32 / (1u64 << 24) as f32 - 0.5)
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
            let q = if scale != 0.0 {
                (v / scale).round().clamp(-128.0, 127.0)
            } else {
                0.0
            };
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

fn assert_bits_eq(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "{what}[{i}]: {g:?} vs {w:?}");
    }
}

fn download_bytes(cuda: &Cuda, src: u64, n: usize) -> Vec<u8> {
    assert_eq!(n % 4, 0, "test byte downloads use an f32-sized region");
    let mut words = vec![0f32; n / 4];
    cuda.dtoh_f32(&mut words, src).unwrap();
    // SAFETY: every initialized f32 bit pattern is valid to view as bytes;
    // the returned Vec owns a copy made before `words` is dropped.
    unsafe { std::slice::from_raw_parts(words.as_ptr().cast::<u8>(), n) }.to_vec()
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

        let mut buf = BackendBuffer::new(
            &cuda,
            ((out_dim * in_dim + in_dim + out_dim) * 4 + 4096) as u64,
        )
        .unwrap();
        let dw = upload(&cuda, &mut buf, &w);
        let dx = upload(&cuda, &mut buf, &x);
        let dy = buf.alloc_f32(out_dim).unwrap().dptr;
        k.gemv(&cuda, dw, dx, dy, out_dim as u32, in_dim as u32)
            .unwrap();
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
    k.quantize_q8(&cuda, dx, d_qs, d_scales, in_dim as u32)
        .unwrap();
    let dy = buf.alloc_f32(out_dim).unwrap().dptr;
    k.gemv_q8_0(&cuda, dw, d_qs, d_scales, dy, out_dim as u32, in_dim as u32)
        .unwrap();
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
    k.quantize_q8(&cuda, dx, d_qs, d_scales, in_dim as u32)
        .unwrap();
    let dy = buf.alloc_f32(out_dim).unwrap().dptr;
    k.gemv_q8_0_soa(
        &cuda,
        dwqs,
        dwsc,
        d_qs,
        d_scales,
        dy,
        out_dim as u32,
        in_dim as u32,
    )
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
    k.quantize_q8(&cuda, dx, d_qs, d_scales, in_dim as u32)
        .unwrap();
    let dy = buf.alloc_f32(out_dim).unwrap().dptr;
    k.gemv_q4_k_soa(
        &cuda,
        dwqs,
        dwsc,
        dwmn,
        d_qs,
        d_scales,
        dy,
        out_dim as u32,
        in_dim as u32,
    )
    .unwrap();
    cuda.synchronize().unwrap();
    let mut got = vec![0f32; out_dim];
    cuda.dtoh_f32(&mut got, dy).unwrap();
    buf.free(&cuda).unwrap();

    assert_close(&got, &want, EPS_Q4K_GEMV, "gemv_q4_k_soa");
}

/// M2.2 Task C-1: the SoA Q6_K GEMV vs the glproc scalar ground truth.
/// All four weight streams are lossless (verbatim i8/f16 scales, relocated
/// quants — no premultiply), so the error structure is the Q8_0 test's:
/// activation quantization (mirrored in the reference via q8_round_trip)
/// plus accumulation order. 2e-3 doubles the Q8 epsilon for the 4x larger
/// integer dots over the 512-wide rows — still 5x under the task's 1e-2.
#[test]
fn gemv_q6_k_soa_matches_dequantized_reference() {
    let Some((cuda, k)) = gpu() else { return };
    let (out_dim, in_dim) = (48usize, 512usize); // 2 super-blocks per row
    let blocks_len = (in_dim / 256 * 210) * out_dim;
    let mut blocks = randv_bytes(blocks_len, 70);
    for block in blocks.chunks_exact_mut(210) {
        block[208..210].copy_from_slice(&0x1e66u16.to_le_bytes()); // sane d
    }
    let w_deq = glproc::kernels::dequant::q6_k::scalar::run(&blocks).unwrap();
    let x = randv(in_dim, 71, 1.0);
    let x_dq = q8_round_trip(&x);
    let mut want = vec![0f32; out_dim];
    glproc::kernels::matmul::scalar::run_matvec(&w_deq, &x_dq, &mut want, out_dim, in_dim);

    let (wql, wqh, wsc, wd) = glcuda::repack::q6_k_to_soa(&blocks).unwrap();

    let bytes =
        (wql.len() + wqh.len() + wsc.len() + wd.len() + (in_dim + out_dim) * 4 + 8192) as u64;
    let mut buf = BackendBuffer::new(&cuda, bytes).unwrap();
    let dql = buf.alloc(wql.len() as u64).unwrap().dptr;
    cuda.htod(dql, &wql).unwrap();
    let dqh = buf.alloc(wqh.len() as u64).unwrap().dptr;
    cuda.htod(dqh, &wqh).unwrap();
    let dsc = buf.alloc(wsc.len() as u64).unwrap().dptr;
    cuda.htod(dsc, &wsc).unwrap();
    let dd = buf.alloc(wd.len() as u64).unwrap().dptr;
    cuda.htod(dd, &wd).unwrap();
    let dx = upload(&cuda, &mut buf, &x);
    let d_qs = buf.alloc(in_dim as u64).unwrap().dptr;
    let d_scales = buf.alloc_f32(in_dim / 32).unwrap().dptr;
    k.quantize_q8(&cuda, dx, d_qs, d_scales, in_dim as u32)
        .unwrap();
    let dy = buf.alloc_f32(out_dim).unwrap().dptr;
    k.gemv_q6_k_soa(
        &cuda,
        dql,
        dqh,
        dsc,
        dd,
        d_qs,
        d_scales,
        dy,
        out_dim as u32,
        in_dim as u32,
    )
    .unwrap();
    cuda.synchronize().unwrap();
    let mut got = vec![0f32; out_dim];
    cuda.dtoh_f32(&mut got, dy).unwrap();
    buf.free(&cuda).unwrap();

    // 2e-3: Q8-like error structure, doubled for the larger q6 dot range.
    assert_close(&got, &want, 2e-3, "gemv_q6_k_soa");
}

/// M2.2 Task C-2: the SoA Q4_0 GEMV vs the glproc scalar ground truth.
/// (48, 384) exercises one full 256-value group + a 4-block tail; (20, 160)
/// is tail-only (dim-896-class models never hit the grouped path evenly).
/// Scales are verbatim f16 (no premul), so the only error sources are the
/// int8 activation quantization (in the reference via q8_round_trip) and
/// accumulation order — the same structure as the Q8_0 SoA test, hence the
/// same 1e-3 epsilon (stricter than the task's 1e-2 bound).
#[test]
fn gemv_q4_0_soa_matches_dequantized_reference() {
    let Some((cuda, k)) = gpu() else { return };
    for (out_dim, in_dim, seed) in [(48usize, 384usize, 60u64), (20, 160, 61)] {
        let blocks_len = (in_dim / 32 * 18) * out_dim;
        let mut blocks = randv_bytes(blocks_len, seed);
        for block in blocks.chunks_exact_mut(18) {
            block[0..2].copy_from_slice(&0x2e66u16.to_le_bytes()); // ~0.1
        }
        let w_deq = glproc::kernels::dequant::q4_0::scalar::run(&blocks);
        let x = randv(in_dim, seed + 1, 1.0);
        let x_dq = q8_round_trip(&x);
        let mut want = vec![0f32; out_dim];
        glproc::kernels::matmul::scalar::run_matvec(&w_deq, &x_dq, &mut want, out_dim, in_dim);

        let (wqs, wsc) = glcuda::repack::q4_0_to_soa(&blocks).unwrap();

        let mut buf = BackendBuffer::new(
            &cuda,
            (wqs.len() + wsc.len() + (in_dim + out_dim) * 4 + 8192) as u64,
        )
        .unwrap();
        let dwqs = buf.alloc(wqs.len() as u64).unwrap().dptr;
        cuda.htod(dwqs, &wqs).unwrap();
        let dwsc = buf.alloc(wsc.len() as u64).unwrap().dptr;
        cuda.htod(dwsc, &wsc).unwrap();
        let dx = upload(&cuda, &mut buf, &x);
        let d_qs = buf.alloc(in_dim as u64).unwrap().dptr;
        let d_scales = buf.alloc_f32(in_dim / 32).unwrap().dptr;
        k.quantize_q8(&cuda, dx, d_qs, d_scales, in_dim as u32)
            .unwrap();
        let dy = buf.alloc_f32(out_dim).unwrap().dptr;
        k.gemv_q4_0_soa(
            &cuda,
            dwqs,
            dwsc,
            d_qs,
            d_scales,
            dy,
            out_dim as u32,
            in_dim as u32,
        )
        .unwrap();
        cuda.synchronize().unwrap();
        let mut got = vec![0f32; out_dim];
        cuda.dtoh_f32(&mut got, dy).unwrap();
        buf.free(&cuda).unwrap();

        assert_close(
            &got,
            &want,
            EPS_Q8_GEMV,
            &format!("gemv_q4_0_soa({out_dim}x{in_dim})"),
        );
    }
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
    k.gemv_q4_0(&cuda, dw, dx, dy, out_dim as u32, in_dim as u32)
        .unwrap();
    cuda.synchronize().unwrap();
    let mut got = vec![0f32; out_dim];
    cuda.dtoh_f32(&mut got, dy).unwrap();
    buf.free(&cuda).unwrap();

    assert_close(&got, &want, EPS_MATMUL, "gemv_q4_0");
}

/// M2.1 Task B / M2.3 Stage 2a: the tensor-core batched GEMM against the
/// same dequantized reference as the dp4a GEMM. Runs only on sm_75+ (the
/// module is not even loaded below that). Ragged token counts exercise the
/// padded-row read / guarded-write contract; 5 stays inside m-tile 0, 20
/// spans three m-tiles of the Stage 2a k-outer loop (weight fragment reused
/// from registers across tiles), 64 fills all eight.
#[test]
fn gemm_mma_q8_matches_dequantized_reference() {
    let Some((cuda, k)) = gpu() else { return };
    if !k.has_mma() {
        eprintln!("SKIP: device below sm_75 — no tensor-core module");
        return;
    }
    for (out_dim, in_dim, ntok) in [(16usize, 64usize, 5usize), (16, 64, 20), (16, 64, 64)] {
        gemm_mma_case(&cuda, &k, out_dim, in_dim, ntok, false);
    }
    for (out_dim, in_dim, ntok) in REAL_SHAPES {
        gemm_mma_case(&cuda, &k, out_dim, in_dim, ntok, false);
    }
}

/// Wave 3: the 8-m-tile kernel now spans token slabs through grid.y instead
/// of a serial host loop. These cases cross the 64-row CTA boundary, include
/// ragged tails, and pin the non-power-of-two dim-896 production geometry.
#[test]
fn gemm_mma_q8_grid2d_matches_dequantized_reference() {
    let Some((cuda, k)) = gpu() else { return };
    if !k.has_mma() {
        eprintln!("SKIP: device below sm_75 — no tensor-core module");
        return;
    }
    for (out_dim, in_dim, ntok) in [
        (16usize, 64usize, 65usize),
        (16, 64, 200),
        (16, 64, 256),
        (256, 896, 200),
    ] {
        gemm_mma_case(&cuda, &k, out_dim, in_dim, ntok, false);
    }
}

/// Wave 12 keeps the arithmetic contract unchanged while changing both CTA
/// width and the physical B image. Prove N64 and N128 bit-for-bit against the
/// retained direct-B kernel, including a ragged final N tile and a second,
/// one-row token slab. This is stricter than the scalar tolerance tests above.
/// Wave 17: prefetching the next k-block must change WHEN loads happen and
/// nothing else.
///
/// The pipelined kernel issues its four global loads one k-block early so their
/// latency lands under the previous block's arithmetic — Wave 16B measured that
/// exposed latency at **25.9%** of the retained kernel, the largest non-
/// arithmetic thing in it. Operand order, accumulation order and the MMAs
/// themselves are untouched, so the output must be **bit-identical**, and that
/// is the only reason a timing number from it would mean anything.
///
/// The shapes pin the pipeline's edges: a single k-block, where the loop's
/// prefetch is predicated off immediately and only the prologue runs; two
/// blocks, the shortest case with a real steady state; and a longer one.
#[test]
fn wave17_pipelined_bstage_is_bit_exact_to_retained_bstage() {
    let Some((cuda, k)) = gpu() else { return };
    if !k.has_mma() {
        eprintln!("SKIP: device below sm_75 — no tensor-core module");
        return;
    }
    for (out_dim, in_dim, ntok) in [
        // One k-block: the loop never prefetches, so only the prologue feeds it.
        (72usize, 32usize, 65usize),
        (72, 64, 65),
        (136, 256, 65),
        // A single token row, where most m-tiles sit idle.
        (72, 128, 1),
    ] {
        wave17_pipe_case(&cuda, &k, out_dim, in_dim, ntok);
    }
}

fn wave17_pipe_case(cuda: &Cuda, k: &KernelSet, out_dim: usize, in_dim: usize, ntok: usize) {
    let ntok_pad = ntok.div_ceil(8) * 8;
    let w_f32 = randv(out_dim * in_dim, 170 + in_dim as u64, 0.1);
    let blocks = glproc::kernels::dequant::q8_0::scalar::quantize(&w_f32);
    let mut qs = Vec::with_capacity(out_dim * in_dim);
    let mut scales = Vec::with_capacity(out_dim * (in_dim / 32) * 2);
    for block in blocks.chunks_exact(34) {
        scales.extend_from_slice(&block[..2]);
        qs.extend_from_slice(&block[2..]);
    }
    let (tiled_qs, tiled_scales) = q8_0_soa_to_bstage(&qs, &scales, out_dim, in_dim).unwrap();
    let x = randv(ntok_pad * in_dim, 180 + in_dim as u64, 1.0);

    let bytes = (tiled_qs.len()
        + tiled_scales.len()
        + ntok_pad * in_dim * 4
        + ntok_pad * in_dim
        + ntok_pad * (in_dim / 32) * 4
        + 2 * ntok * out_dim * 4
        + 64 * 1024) as u64;
    let mut buf = BackendBuffer::new(cuda, bytes).unwrap();
    let dtqs = buf.alloc(tiled_qs.len() as u64).unwrap().dptr;
    let dtsc = buf.alloc(tiled_scales.len() as u64).unwrap().dptr;
    cuda.htod(dtqs, &tiled_qs).unwrap();
    cuda.htod(dtsc, &tiled_scales).unwrap();
    let dx = upload(cuda, &mut buf, &x);
    let dxqs = buf.alloc((ntok_pad * in_dim) as u64).unwrap().dptr;
    let dxsc = buf.alloc_f32(ntok_pad * in_dim / 32).unwrap().dptr;
    k.quantize_q8(cuda, dx, dxqs, dxsc, (ntok_pad * in_dim) as u32)
        .unwrap();
    let retained = buf.alloc_f32(ntok * out_dim).unwrap().dptr;
    let pipelined = buf.alloc_f32(ntok * out_dim).unwrap().dptr;

    k.gemm_mma_q8_bstage(
        cuda,
        dtqs,
        dtsc,
        dxqs,
        dxsc,
        retained,
        out_dim as u32,
        in_dim as u32,
        ntok as u32,
    )
    .unwrap();
    k.gemm_mma_q8_bstage_pipe(
        cuda,
        dtqs,
        dtsc,
        dxqs,
        dxsc,
        pipelined,
        out_dim as u32,
        in_dim as u32,
        ntok as u32,
    )
    .unwrap();
    cuda.synchronize().unwrap();

    let mut a = vec![0f32; ntok * out_dim];
    let mut b = vec![0f32; ntok * out_dim];
    cuda.dtoh_f32(&mut a, retained).unwrap();
    cuda.dtoh_f32(&mut b, pipelined).unwrap();
    buf.free(cuda).unwrap();
    assert_bits_eq(
        &b,
        &a,
        &format!("Wave17 pipelined bstage out={out_dim} in={in_dim} ntok={ntok}"),
    );
}

#[test]
fn gemm_mma_q8_bstage_is_bit_exact_at_both_n_tiles() {
    let Some((cuda, k)) = gpu() else { return };
    if !k.has_mma() {
        eprintln!("SKIP: device below sm_75 — no tensor-core module");
        return;
    }
    for (full_out, row0, rows, n_tile) in [
        (72usize, 0usize, 72usize, 64u32),
        (136, 0, 136, 128),
        // The production gate/up matrix is stacked and its up projection
        // starts at a non-zero but N128-aligned row tile.
        (272, 128, 136, 128),
    ] {
        gemm_mma_bstage_exact_case(&cuda, &k, full_out, row0, rows, 64, 65, n_tile);
    }
}

#[allow(clippy::too_many_arguments)]
fn gemm_mma_bstage_exact_case(
    cuda: &Cuda,
    k: &KernelSet,
    full_out: usize,
    row0: usize,
    rows: usize,
    in_dim: usize,
    ntok: usize,
    n_tile: u32,
) {
    let ntok_pad = ntok.div_ceil(8) * 8;
    let w_f32 = randv(full_out * in_dim, 70 + n_tile as u64 + row0 as u64, 0.1);
    let blocks = glproc::kernels::dequant::q8_0::scalar::quantize(&w_f32);
    let mut qs = Vec::with_capacity(full_out * in_dim);
    let mut scales = Vec::with_capacity(full_out * (in_dim / 32) * 2);
    for block in blocks.chunks_exact(34) {
        scales.extend_from_slice(&block[..2]);
        qs.extend_from_slice(&block[2..]);
    }
    let (tiled_qs, tiled_scales) = q8_0_soa_to_bstage(&qs, &scales, full_out, in_dim).unwrap();
    let x = randv(ntok_pad * in_dim, 80 + n_tile as u64, 1.0);

    let bytes = (qs.len()
        + scales.len()
        + tiled_qs.len()
        + tiled_scales.len()
        + ntok_pad * in_dim * 4
        + ntok_pad * in_dim
        + ntok_pad * (in_dim / 32) * 4
        + 2 * ntok * rows * 4
        + 64 * 1024) as u64;
    let mut buf = BackendBuffer::new(cuda, bytes).unwrap();
    let dwqs = buf.alloc(qs.len() as u64).unwrap().dptr;
    let dwsc = buf.alloc(scales.len() as u64).unwrap().dptr;
    let dtqs = buf.alloc(tiled_qs.len() as u64).unwrap().dptr;
    let dtsc = buf.alloc(tiled_scales.len() as u64).unwrap().dptr;
    cuda.htod(dwqs, &qs).unwrap();
    cuda.htod(dwsc, &scales).unwrap();
    cuda.htod(dtqs, &tiled_qs).unwrap();
    cuda.htod(dtsc, &tiled_scales).unwrap();
    let dx = upload(cuda, &mut buf, &x);
    let dxqs = buf.alloc((ntok_pad * in_dim) as u64).unwrap().dptr;
    let dxsc = buf.alloc_f32(ntok_pad * in_dim / 32).unwrap().dptr;
    k.quantize_q8(cuda, dx, dxqs, dxsc, (ntok_pad * in_dim) as u32)
        .unwrap();
    let direct = buf.alloc_f32(ntok * rows).unwrap().dptr;
    let staged = buf.alloc_f32(ntok * rows).unwrap().dptr;
    let nb = in_dim / 32;
    let direct_qs = dwqs + (row0 * in_dim) as u64;
    let direct_scales = dwsc + (row0 * nb * 2) as u64;
    let tile0 = row0 / 128;
    let staged_qs = dtqs + (tile0 * nb * 128 * 32) as u64;
    let staged_scales = dtsc + (tile0 * nb * 128 * 2) as u64;

    k.gemm_mma_q8_diagnostic_ntile(
        cuda,
        direct_qs,
        direct_scales,
        dxqs,
        dxsc,
        direct,
        rows as u32,
        in_dim as u32,
        ntok as u32,
        n_tile,
    )
    .unwrap();
    k.gemm_mma_q8_bstage_diagnostic(
        cuda,
        staged_qs,
        staged_scales,
        dxqs,
        dxsc,
        staged,
        rows as u32,
        in_dim as u32,
        ntok as u32,
        n_tile,
    )
    .unwrap();
    cuda.synchronize().unwrap();
    let mut direct_h = vec![0f32; ntok * rows];
    let mut staged_h = vec![0f32; ntok * rows];
    cuda.dtoh_f32(&mut direct_h, direct).unwrap();
    cuda.dtoh_f32(&mut staged_h, staged).unwrap();
    buf.free(cuda).unwrap();
    assert_bits_eq(
        &staged_h,
        &direct_h,
        &format!("Wave12 B-stage N{n_tile} row0={row0} exact MMA output"),
    );
}

/// Wave 12 factor A launches the *retained* direct kernel with 512 threads so
/// one CTA owns 128 output columns instead of 64. No arithmetic changes: each
/// 8-column tile is still one warp walking the same k-loop, only the warp's
/// address changes. So N128 must reproduce the trusted N64 launch bit-for-bit
/// at the shapes production actually runs.
///
/// This is the check [`gemm_mma_q8_bstage_is_bit_exact_at_both_n_tiles`]
/// cannot make. That one compares two Wave 12 paths against each other, and a
/// defect in the shared `ntid.x`-derived tile mapping would appear in both.
#[test]
fn gemm_mma_q8_ntile128_is_bit_exact_to_the_retained_n64_launch() {
    let Some((cuda, k)) = gpu() else { return };
    if !k.has_mma() {
        eprintln!("SKIP: device below sm_75 — no tensor-core module");
        return;
    }
    for (out_dim, in_dim, ntok) in REAL_SHAPES {
        gemm_mma_ntile_exact_case(&cuda, &k, out_dim, in_dim, ntok);
    }
    // The pinned prompt shape: 244 tokens is four grid.y slabs with a ragged
    // tail, and 4864 outputs is where the N128 coverage guard actually fires.
    gemm_mma_ntile_exact_case(&cuda, &k, 4_864, 896, 244);
    // Ragged N tile: 72 output rows leave seven of sixteen N128 warps out of
    // range, which is the inactive-warp staging path.
    gemm_mma_ntile_exact_case(&cuda, &k, 72, 64, 65);
}

fn gemm_mma_ntile_exact_case(
    cuda: &Cuda,
    k: &KernelSet,
    out_dim: usize,
    in_dim: usize,
    ntok: usize,
) {
    let ntok_pad = ntok.div_ceil(8) * 8;
    let nb = in_dim / 32;
    let w_f32 = randv(out_dim * in_dim, 90 + out_dim as u64, 0.1);
    let blocks = glproc::kernels::dequant::q8_0::scalar::quantize(&w_f32);
    let mut qs = Vec::with_capacity(out_dim * in_dim);
    let mut scales = Vec::with_capacity(out_dim * nb * 2);
    for block in blocks.chunks_exact(34) {
        scales.extend_from_slice(&block[..2]);
        qs.extend_from_slice(&block[2..]);
    }
    let x = randv(ntok_pad * in_dim, 91 + ntok as u64, 1.0);

    let bytes = (qs.len()
        + scales.len()
        + ntok_pad * in_dim * 4
        + ntok_pad * in_dim
        + ntok_pad * nb * 4
        + 2 * ntok * out_dim * 4
        + 64 * 1024) as u64;
    let mut buf = BackendBuffer::new(cuda, bytes).unwrap();
    let dwqs = buf.alloc(qs.len() as u64).unwrap().dptr;
    let dwsc = buf.alloc(scales.len() as u64).unwrap().dptr;
    cuda.htod(dwqs, &qs).unwrap();
    cuda.htod(dwsc, &scales).unwrap();
    let dx = upload(cuda, &mut buf, &x);
    let dxqs = buf.alloc((ntok_pad * in_dim) as u64).unwrap().dptr;
    let dxsc = buf.alloc_f32(ntok_pad * nb).unwrap().dptr;
    k.quantize_q8(cuda, dx, dxqs, dxsc, (ntok_pad * in_dim) as u32)
        .unwrap();
    let n64 = buf.alloc_f32(ntok * out_dim).unwrap().dptr;
    let n128 = buf.alloc_f32(ntok * out_dim).unwrap().dptr;

    for (y, n_tile) in [(n64, 64u32), (n128, 128u32)] {
        k.gemm_mma_q8_diagnostic_ntile(
            cuda,
            dwqs,
            dwsc,
            dxqs,
            dxsc,
            y,
            out_dim as u32,
            in_dim as u32,
            ntok as u32,
            n_tile,
        )
        .unwrap();
    }
    cuda.synchronize().unwrap();
    let mut n64_h = vec![0f32; ntok * out_dim];
    let mut n128_h = vec![0f32; ntok * out_dim];
    cuda.dtoh_f32(&mut n64_h, n64).unwrap();
    cuda.dtoh_f32(&mut n128_h, n128).unwrap();
    buf.free(cuda).unwrap();
    assert_bits_eq(
        &n128_h,
        &n64_h,
        &format!("Wave12 direct N128 out={out_dim} in={in_dim} ntok={ntok} vs retained N64"),
    );
}

/// Wave 13B: the whole post-projection chain must not care whether Q/K/V are
/// packed buffers or column slices of one stacked slab.
///
/// Bias, RoPE, the KV write and attention all used to derive the distance
/// between token rows from their own row width, which is what made the layout
/// unchangeable. Each now takes that distance. This runs the real production
/// order twice over identical values, once packed and once strided at the
/// Qwen2.5-0.5B stacked width, and demands bit-identical attention output.
#[test]
fn strided_projection_slices_match_the_packed_layout_through_attention() {
    let Some((cuda, k)) = gpu() else { return };
    let (head_dim, n_heads, n_kv, ntok, max_ctx) = (64usize, 14usize, 2usize, 20usize, 64usize);
    let (q_dim, kv_dim) = (n_heads * head_dim, n_kv * head_dim);
    let stacked = q_dim + 2 * kv_dim;
    let heads_per_kv = (n_heads / n_kv) as u32;
    let head_stride = max_ctx * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let q_src = randv(ntok * q_dim, 120, 1.0);
    let k_src = randv(ntok * kv_dim, 121, 1.0);
    let v_src = randv(ntok * kv_dim, 122, 1.0);
    let q_bias = randv(q_dim, 123, 0.5);
    let k_bias = randv(kv_dim, 124, 0.5);
    let cos = randv(max_ctx * head_dim / 2, 125, 1.0);
    let sin = randv(max_ctx * head_dim / 2, 126, 1.0);

    // Interleave the same values into one wide slab: [q | k | v] per row.
    let mut slab = vec![0f32; ntok * stacked];
    for t in 0..ntok {
        let row = &mut slab[t * stacked..(t + 1) * stacked];
        row[..q_dim].copy_from_slice(&q_src[t * q_dim..(t + 1) * q_dim]);
        row[q_dim..q_dim + kv_dim].copy_from_slice(&k_src[t * kv_dim..(t + 1) * kv_dim]);
        row[q_dim + kv_dim..].copy_from_slice(&v_src[t * kv_dim..(t + 1) * kv_dim]);
    }

    let bytes = ((ntok * stacked
        + ntok * (q_dim + 2 * kv_dim)
        + 2 * ntok * q_dim
        + 4 * n_kv * head_stride
        + max_ctx * head_dim
        + q_dim
        + kv_dim)
        * 4
        + (max_ctx + 1) * 4
        + 128 * 1024) as u64;
    let mut buf = BackendBuffer::new(&cuda, bytes).unwrap();
    let pos_vals: Vec<u32> = (0..=(max_ctx as u32)).collect();
    let dpos = buf.alloc(((max_ctx + 1) * 4) as u64).unwrap().dptr;
    let pos_bytes =
        unsafe { std::slice::from_raw_parts(pos_vals.as_ptr().cast::<u8>(), pos_vals.len() * 4) };
    cuda.htod(dpos, pos_bytes).unwrap();
    let dcos = upload(&cuda, &mut buf, &cos);
    let dsin = upload(&cuda, &mut buf, &sin);
    let dqb = upload(&cuda, &mut buf, &q_bias);
    let dkb = upload(&cuda, &mut buf, &k_bias);

    // Run the production order for one layout and return the attention output.
    let run = |q: u64, kk: u64, v: u64, qs: u32, kvs: u32, buf: &mut BackendBuffer| {
        let kc = buf.alloc_f32(n_kv * head_stride).unwrap().dptr;
        let vc = buf.alloc_f32(n_kv * head_stride).unwrap().dptr;
        let out = buf.alloc_f32(ntok * q_dim).unwrap().dptr;
        k.add_bias_rows(&cuda, q, dqb, q_dim as u32, (ntok * q_dim) as u32, qs)
            .unwrap();
        k.add_bias_rows(&cuda, kk, dkb, kv_dim as u32, (ntok * kv_dim) as u32, kvs)
            .unwrap();
        k.rope_rows(
            &cuda,
            q,
            dcos,
            dsin,
            n_heads as u32,
            head_dim as u32,
            false,
            dpos,
            ntok as u32,
            qs,
        )
        .unwrap();
        k.rope_rows(
            &cuda,
            kk,
            dcos,
            dsin,
            n_kv as u32,
            head_dim as u32,
            false,
            dpos,
            ntok as u32,
            kvs,
        )
        .unwrap();
        k.kv_write_rows(
            &cuda,
            kc,
            kk,
            dpos,
            head_dim as u32,
            n_kv as u32,
            head_stride as u32,
            ntok as u32,
            kvs,
        )
        .unwrap();
        k.kv_write_rows(
            &cuda,
            vc,
            v,
            dpos,
            head_dim as u32,
            n_kv as u32,
            head_stride as u32,
            ntok as u32,
            kvs,
        )
        .unwrap();
        k.attn_decode_rows_legacy(
            &cuda,
            q,
            kc,
            vc,
            out,
            n_heads as u32,
            head_dim as u32,
            dpos,
            heads_per_kv,
            head_stride as u32,
            scale,
            ntok as u32,
            ntok as u32,
            qs,
        )
        .unwrap();
        cuda.synchronize().unwrap();
        let mut host = vec![0f32; ntok * q_dim];
        cuda.dtoh_f32(&mut host, out).unwrap();
        host
    };

    let dq = upload(&cuda, &mut buf, &q_src);
    let dk = upload(&cuda, &mut buf, &k_src);
    let dv = upload(&cuda, &mut buf, &v_src);
    let packed = run(dq, dk, dv, q_dim as u32, kv_dim as u32, &mut buf);

    let dslab = upload(&cuda, &mut buf, &slab);
    let strided = run(
        dslab,
        dslab + (q_dim * 4) as u64,
        dslab + ((q_dim + kv_dim) * 4) as u64,
        stacked as u32,
        stacked as u32,
        &mut buf,
    );
    buf.free(&cuda).unwrap();
    assert_bits_eq(
        &strided,
        &packed,
        "Wave13B stacked-slab chain vs packed chain",
    );
}

/// Wave 15A is retained, so the DEFAULT path is the four-chain QK.
///
/// This is the whole point of the retention: with no environment set at all, a
/// plain embedder gets the kernel that measured +8.59% over the row kernel and
/// +3.65% over GQA7 in production. `GLCUDA_ATTN_ROWS` is the way back, and an
/// A/B is the only thing that should want it.
///
/// Switching a default is only safe because every one of these kernels is
/// bit-identical to the others - proven by the three tests above this one - so
/// this changes the schedule and not one output bit.
#[test]
fn wave15a_four_chain_qk_is_the_default_path() {
    let Some((_cuda, k)) = gpu() else { return };
    // The production shape: 14 heads over 2 KV heads, head_dim 64.
    let call = glcuda::attention::VLAttentionCall {
        n_tokens: 244,
        pos_base: 0,
        n_heads: 14,
        n_kv_heads: 2,
        head_dim: 64,
        head_stride: 256 * 64,
        scale: 0.125,
    };
    let path = glcuda::attention::select(&k, &call);
    assert_eq!(
        path,
        glcuda::attention::ENAttentionPath::Qk4,
        "the retained four-chain QK must be what an unconfigured engine runs"
    );
}

/// Wave 15D: halving the K/V tile must move the shared-memory budget and
/// nothing else.
///
/// Warp `w` owned rows `w` and `w+4` of an eight-row tile; with four-row tiles
/// it owns row `w` of two consecutive ones. Same rows, same order, so the
/// ascending-t accumulation is unchanged and the output must be bit-identical.
/// That matters more than usual here: the wave's entire claim is that only the
/// allocation moved, and a numeric difference would mean it moved something
/// else too.
#[test]
fn wave15d_four_row_tile_is_bit_exact_to_retained_gqa7() {
    let Some((cuda, k)) = gpu() else { return };
    for (ntok, base, max_ctx) in [
        (8usize, 236usize, 288usize),
        // A history that is not a multiple of either tile height.
        (13, 232, 288),
        // Fewer keys than one four-row tile, so most rows read the zero fill.
        (3, 0, 64),
        // Exactly one eight-row tile, which is two four-row ones.
        (8, 0, 64),
    ] {
        wave15d_case(&cuda, &k, ntok, base, max_ctx);
    }
}

fn wave15d_case(cuda: &Cuda, k: &KernelSet, ntok: usize, base: usize, max_ctx: usize) {
    let (head_dim, n_heads, n_kv) = (64usize, 14usize, 2usize);
    let filled = base + ntok;
    let heads_per_kv = 7usize;
    let head_stride = max_ctx * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut kc = vec![0f32; n_kv * head_stride];
    let mut vc = vec![0f32; n_kv * head_stride];
    for kvh in 0..n_kv {
        let sk = randv(filled * head_dim, 0x15d0 + kvh as u64, 1.0);
        let sv = randv(filled * head_dim, 0x15e0 + kvh as u64, 1.0);
        kc[kvh * head_stride..kvh * head_stride + sk.len()].copy_from_slice(&sk);
        vc[kvh * head_stride..kvh * head_stride + sv.len()].copy_from_slice(&sv);
    }
    let q = randv(ntok * n_heads * head_dim, 0x15f0 + ntok as u64, 1.0);
    let bytes = ((q.len() * 3 + kc.len() + vc.len()) * 4 + (max_ctx + 1) * 4 + 32_768) as u64;
    let mut buf = BackendBuffer::new(cuda, bytes).unwrap();
    let dq = upload(cuda, &mut buf, &q);
    let dk = upload(cuda, &mut buf, &kc);
    let dv = upload(cuda, &mut buf, &vc);
    let retained = buf.alloc_f32(q.len()).unwrap().dptr;
    let tiled = buf.alloc_f32(q.len()).unwrap().dptr;
    let pos_vals: Vec<u32> = (0..=(max_ctx as u32)).collect();
    let dpos = buf.alloc(((max_ctx + 1) * 4) as u64).unwrap().dptr;
    let pos_bytes =
        unsafe { std::slice::from_raw_parts(pos_vals.as_ptr().cast::<u8>(), pos_vals.len() * 4) };
    cuda.htod(dpos, pos_bytes).unwrap();
    let pos_base = dpos + (base * 4) as u64;
    let q_stride = (n_heads * head_dim) as u32;

    k.attn_decode_rows_gqa7(
        cuda,
        dq,
        dk,
        dv,
        retained,
        n_heads as u32,
        head_dim as u32,
        pos_base,
        heads_per_kv as u32,
        head_stride as u32,
        scale,
        ntok as u32,
        filled as u32,
        q_stride,
    )
    .unwrap();
    k.attn_gqa7_t4(
        cuda,
        dq,
        dk,
        dv,
        tiled,
        n_heads as u32,
        head_dim as u32,
        pos_base,
        heads_per_kv as u32,
        head_stride as u32,
        scale,
        ntok as u32,
        filled as u32,
        q_stride,
    )
    .unwrap();
    cuda.synchronize().unwrap();

    let mut a = vec![0f32; q.len()];
    let mut b = vec![0f32; q.len()];
    cuda.dtoh_f32(&mut a, retained).unwrap();
    cuda.dtoh_f32(&mut b, tiled).unwrap();
    buf.free(cuda).unwrap();
    assert_bits_eq(
        &b,
        &a,
        &format!("Wave15D gqa7 4-row tile ntok={ntok} base={base} max_ctx={max_ctx}"),
    );
}

/// Wave 15B: chaining GQA7 must change the schedule and nothing else.
///
/// Both chained kernels reduce every score in exactly the retained GQA7 order
/// over exactly its operands and read exactly its eight-row K tile, so this is
/// bit-exactness rather than a tolerance. The factorial depends on it: if the
/// candidates disagreed numerically, a throughput difference could be a
/// different computation rather than a different schedule.
///
/// The shapes pin the tail arithmetic, where a tile straddles the end of the
/// cache, and the odd seventh head, which QK4 routes through its two-chain
/// block because it has no partner to pair with.
#[test]
fn wave15b_chained_gqa7_is_bit_exact_to_retained_gqa7() {
    let Some((cuda, k)) = gpu() else { return };
    for (ntok, base, max_ctx) in [
        (8usize, 236usize, 288usize),
        // cached_len not a multiple of the eight-row tile.
        (13, 232, 288),
        // A single query row against a long history.
        (1, 243, 256),
        // Fewer keys than one tile, so most rows read the tile's zero fill.
        (3, 0, 64),
    ] {
        for chains in [2u8, 4u8] {
            wave15b_case(&cuda, &k, ntok, base, max_ctx, chains);
        }
    }
}

fn wave15b_case(cuda: &Cuda, k: &KernelSet, ntok: usize, base: usize, max_ctx: usize, chains: u8) {
    let (head_dim, n_heads, n_kv) = (64usize, 14usize, 2usize);
    let filled = base + ntok;
    let heads_per_kv = 7usize;
    let head_stride = max_ctx * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut kc = vec![0f32; n_kv * head_stride];
    let mut vc = vec![0f32; n_kv * head_stride];
    for kvh in 0..n_kv {
        let sk = randv(filled * head_dim, 0x15b0 + kvh as u64, 1.0);
        let sv = randv(filled * head_dim, 0x15c0 + kvh as u64, 1.0);
        kc[kvh * head_stride..kvh * head_stride + sk.len()].copy_from_slice(&sk);
        vc[kvh * head_stride..kvh * head_stride + sv.len()].copy_from_slice(&sv);
    }
    let q = randv(ntok * n_heads * head_dim, 0x15d0 + ntok as u64, 1.0);
    let bytes = ((q.len() * 3 + kc.len() + vc.len()) * 4 + (max_ctx + 1) * 4 + 32_768) as u64;
    let mut buf = BackendBuffer::new(cuda, bytes).unwrap();
    let dq = upload(cuda, &mut buf, &q);
    let dk = upload(cuda, &mut buf, &kc);
    let dv = upload(cuda, &mut buf, &vc);
    let retained = buf.alloc_f32(q.len()).unwrap().dptr;
    let chained = buf.alloc_f32(q.len()).unwrap().dptr;
    let pos_vals: Vec<u32> = (0..=(max_ctx as u32)).collect();
    let dpos = buf.alloc(((max_ctx + 1) * 4) as u64).unwrap().dptr;
    let pos_bytes =
        unsafe { std::slice::from_raw_parts(pos_vals.as_ptr().cast::<u8>(), pos_vals.len() * 4) };
    cuda.htod(dpos, pos_bytes).unwrap();
    let pos_base = dpos + (base * 4) as u64;
    let q_stride = (n_heads * head_dim) as u32;

    k.attn_decode_rows_gqa7(
        cuda,
        dq,
        dk,
        dv,
        retained,
        n_heads as u32,
        head_dim as u32,
        pos_base,
        heads_per_kv as u32,
        head_stride as u32,
        scale,
        ntok as u32,
        filled as u32,
        q_stride,
    )
    .unwrap();
    k.attn_gqa7_chained(
        cuda,
        dq,
        dk,
        dv,
        chained,
        n_heads as u32,
        head_dim as u32,
        pos_base,
        heads_per_kv as u32,
        head_stride as u32,
        scale,
        ntok as u32,
        filled as u32,
        q_stride,
        chains,
        0,
    )
    .unwrap();
    cuda.synchronize().unwrap();

    let mut a = vec![0f32; q.len()];
    let mut b = vec![0f32; q.len()];
    cuda.dtoh_f32(&mut a, retained).unwrap();
    cuda.dtoh_f32(&mut b, chained).unwrap();
    buf.free(cuda).unwrap();
    assert_bits_eq(
        &b,
        &a,
        &format!("Wave15B gqa7+qk{chains} ntok={ntok} base={base} max_ctx={max_ctx}"),
    );
}

/// Wave 15A: four independent QK chains per warp must change the schedule and
/// nothing else.
///
/// Each chain reduces in exactly the retained kernel's order over exactly the
/// retained kernel's operands, so this is not a tolerance test: the outputs
/// must be **bit-identical**. That is the strongest gate available and it needs
/// no epsilon to argue about, which is the whole reason Wave 15A took the f32
/// route instead of the tensor cores.
///
/// The cases pin the tail arithmetic, where a warp's four keys straddle the end
/// of the cache and three of them are clamped to the last valid row.
#[test]
fn wave15_four_chain_qk_is_bit_exact_to_the_retained_attention() {
    let Some((cuda, k)) = gpu() else { return };
    for (n_heads, n_kv, ntok, filled) in [
        // The production shape.
        (14usize, 2usize, 244usize, 244usize),
        // cached_len not a multiple of four: the last warp stores one score of
        // the four it computed.
        (4, 2, 8, 245),
        // Fewer keys than one chain group, so three of four are clamped.
        (2, 1, 3, 2),
        // Exactly one chain group.
        (2, 1, 4, 4),
    ] {
        wave15_qk4_case(&cuda, &k, n_heads, n_kv, ntok, filled);
    }
}

fn wave15_qk4_case(
    cuda: &Cuda,
    k: &KernelSet,
    n_heads: usize,
    n_kv: usize,
    ntok: usize,
    filled: usize,
) {
    let head_dim = 64usize;
    let head_stride = filled.max(ntok).next_power_of_two().max(64) * head_dim;
    let heads_per_kv = (n_heads / n_kv).max(1) as u32;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let base = filled.saturating_sub(ntok);

    let q = randv(ntok * n_heads * head_dim, 140 + ntok as u64, 1.0);
    let kc = randv(n_kv * head_stride, 141 + filled as u64, 1.0);
    let vc = randv(n_kv * head_stride, 142 + filled as u64, 1.0);

    let bytes =
        ((q.len() + kc.len() + vc.len() + 2 * q.len()) * 4 + (filled + 2) * 4 + 64 * 1024) as u64;
    let mut buf = BackendBuffer::new(cuda, bytes).unwrap();
    let dq = upload(cuda, &mut buf, &q);
    let dk = upload(cuda, &mut buf, &kc);
    let dv = upload(cuda, &mut buf, &vc);
    let retained = buf.alloc_f32(q.len()).unwrap().dptr;
    let chained = buf.alloc_f32(q.len()).unwrap().dptr;
    let pos_vals: Vec<u32> = (0..=(filled as u32 + 1)).collect();
    let dpos = buf.alloc(((filled + 2) * 4) as u64).unwrap().dptr;
    let pos_bytes =
        unsafe { std::slice::from_raw_parts(pos_vals.as_ptr().cast::<u8>(), pos_vals.len() * 4) };
    cuda.htod(dpos, pos_bytes).unwrap();
    let pos_base = dpos + (base * 4) as u64;
    let q_stride = (n_heads * head_dim) as u32;

    k.attn_decode_rows_legacy(
        cuda,
        dq,
        dk,
        dv,
        retained,
        n_heads as u32,
        head_dim as u32,
        pos_base,
        heads_per_kv,
        head_stride as u32,
        scale,
        ntok as u32,
        filled as u32,
        q_stride,
    )
    .unwrap();
    k.attn_rows_qk4(
        cuda,
        dq,
        dk,
        dv,
        chained,
        n_heads as u32,
        head_dim as u32,
        pos_base,
        heads_per_kv,
        head_stride as u32,
        scale,
        ntok as u32,
        filled as u32,
        q_stride,
    )
    .unwrap();
    cuda.synchronize().unwrap();

    let mut a = vec![0f32; q.len()];
    let mut b = vec![0f32; q.len()];
    cuda.dtoh_f32(&mut a, retained).unwrap();
    cuda.dtoh_f32(&mut b, chained).unwrap();
    buf.free(cuda).unwrap();
    assert_bits_eq(
        &b,
        &a,
        &format!("Wave15A qk4 heads={n_heads} kv={n_kv} ntok={ntok} filled={filled}"),
    );
}

/// Waves 20/48/78: compensated-f16 MMA attention under the exact production
/// contract (causal positions, strided Q, softmax and AV). Wave 78 replaces
/// only AV with cooperative MMA. All candidates are tolerance-gated against
/// the host oracle, including ragged and non-zero-base shapes.
#[test]
fn fused_mma4_attention_matches_oracle_at_production_and_tail_shapes() {
    let Some((cuda, k)) = gpu() else { return };
    if !k.has_mma() {
        eprintln!("SKIP: device below sm_75 - no tensor-core module");
        return;
    }
    for (n_heads, n_kv, base, ntok, q_pad) in [
        (14usize, 2usize, 0usize, 244usize, 0usize),
        (4, 2, 5, 17, 11),
        (2, 1, 7, 3, 5),
    ] {
        wave20_mma4_attention_case(&cuda, &k, n_heads, n_kv, base, ntok, q_pad);
    }
}

#[allow(clippy::too_many_arguments)]
fn wave20_mma4_attention_case(
    cuda: &Cuda,
    k: &KernelSet,
    n_heads: usize,
    n_kv: usize,
    base: usize,
    ntok: usize,
    q_pad: usize,
) {
    let head_dim = 64usize;
    let filled = base + ntok;
    let head_stride = filled.next_power_of_two().max(64) * head_dim;
    let heads_per_kv = (n_heads / n_kv).max(1);
    let scale = 1.0 / (head_dim as f32).sqrt();
    let width = n_heads * head_dim;
    let q_stride = width + q_pad;
    let q_len = (ntok - 1) * q_stride + width;
    let q = randv(q_len, 200 + ntok as u64, 1.0);
    let kc = randv(n_kv * head_stride, 201 + filled as u64, 1.0);
    let vc = randv(n_kv * head_stride, 202 + filled as u64, 1.0);
    let call = glcuda::attention::VLAttentionCall {
        n_tokens: ntok as u32,
        pos_base: base as u32,
        n_heads: n_heads as u32,
        n_kv_heads: n_kv as u32,
        head_dim: head_dim as u32,
        head_stride: head_stride as u32,
        scale,
    };
    let mut want = vec![0f32; ntok * width];
    glcuda::attention::reference::prefill(&q, q_stride, &kc, &vc, &mut want, &call);

    let bytes = ((q.len() + kc.len() + vc.len() + 3 * want.len()) * 4
        + (filled + 1) * 4
        + 64 * 1024) as u64;
    let mut buf = BackendBuffer::new(cuda, bytes).unwrap();
    let dq = upload(cuda, &mut buf, &q);
    let dk = upload(cuda, &mut buf, &kc);
    let dv = upload(cuda, &mut buf, &vc);
    let dout = buf.alloc_f32(want.len()).unwrap().dptr;
    let dregq = buf.alloc_f32(want.len()).unwrap().dptr;
    let davmma = buf.alloc_f32(want.len()).unwrap().dptr;
    let pos_vals: Vec<u32> = (0..=filled as u32).collect();
    let dpos = buf.alloc((pos_vals.len() * 4) as u64).unwrap().dptr;
    let pos_bytes =
        unsafe { std::slice::from_raw_parts(pos_vals.as_ptr().cast::<u8>(), pos_vals.len() * 4) };
    cuda.htod(dpos, pos_bytes).unwrap();
    k.attn_mma4_fused(
        cuda,
        dq,
        dk,
        dv,
        dout,
        n_heads as u32,
        head_dim as u32,
        dpos + (base * 4) as u64,
        heads_per_kv as u32,
        head_stride as u32,
        scale,
        ntok as u32,
        filled as u32,
        q_stride as u32,
    )
    .unwrap();
    k.attn_mma4_regq_fused(
        cuda,
        dq,
        dk,
        dv,
        dregq,
        n_heads as u32,
        head_dim as u32,
        dpos + (base * 4) as u64,
        heads_per_kv as u32,
        head_stride as u32,
        scale,
        ntok as u32,
        filled as u32,
        q_stride as u32,
    )
    .unwrap();
    k.attn_mma4_regq_avmma_fused(
        cuda,
        dq,
        dk,
        dv,
        davmma,
        n_heads as u32,
        head_dim as u32,
        dpos + (base * 4) as u64,
        heads_per_kv as u32,
        head_stride as u32,
        scale,
        ntok as u32,
        filled as u32,
        q_stride as u32,
    )
    .unwrap();
    cuda.synchronize().unwrap();
    let mut got = vec![0f32; want.len()];
    let mut regq = vec![0f32; want.len()];
    let mut avmma = vec![0f32; want.len()];
    cuda.dtoh_f32(&mut got, dout).unwrap();
    cuda.dtoh_f32(&mut regq, dregq).unwrap();
    cuda.dtoh_f32(&mut avmma, davmma).unwrap();
    buf.free(cuda).unwrap();
    assert_close(
        &got,
        &want,
        EPS_MATMUL,
        &format!(
            "Wave20 MMA4 fused heads={n_heads} kv={n_kv} base={base} ntok={ntok} qpad={q_pad}"
        ),
    );
    assert_bits_eq(
        &regq,
        &got,
        &format!(
            "Wave48 register-Q vs Wave20 heads={n_heads} kv={n_kv} base={base} ntok={ntok} qpad={q_pad}"
        ),
    );
    assert_close(
        &avmma,
        &want,
        EPS_MATMUL,
        &format!(
            "Wave78 compensated-MMA AV heads={n_heads} kv={n_kv} base={base} ntok={ntok} qpad={q_pad}"
        ),
    );
}

/// The shapes this engine actually runs, for Qwen2.5-0.5B (dim 896,
/// intermediate 4864).
///
/// ⛔ Every GEMM case here used to be `out_dim=16, in_dim=64`: one block, two
/// of eight warps in range, a K-loop of two 32-blocks. No shape the model
/// executes was covered by any GEMM test, so both "pass" and "fail" said
/// almost nothing about production behaviour — a gap found only when the r256
/// kernel failed and the failure turned out to be unrepresentative either way.
///
/// The x-axis block counts remain important: 76 for `gate`/`up`, 14 for
/// `down`/`o`/`q`, and 2 for `k`/`v` on a 40-SM T4. Wave 3 multiplies those
/// by `ceil_div(ntok, 64)` through grid.y; these ntok=64 cases deliberately
/// keep grid.y=1 and pin the original launch as a no-op boundary.
///
/// Keeping ntok at one full CTA exercises all eight m-tiles; the separate
/// grid2d test covers cross-CTA and ragged-tail behavior.
const REAL_SHAPES: [(usize, usize, usize); 4] = [
    (896, 896, 64),  // q and o_proj: 14 blocks, square
    (128, 896, 64),  // k and v: 2 blocks — the narrowest output in the model
    (4864, 896, 64), // gate and up: 76 blocks, the wide output
    (896, 4864, 64), // down: 14 blocks over the longest K in the model
];

/// Phase B r256 kernel: same math as the 8-tile GEMM but 32 m-tiles (256
/// rows/read). ntok cases span the guard boundaries — 5 (tile 0 only), 128
/// (16 tiles, the A/B's low point), 256 (all 32 tiles, the cap).
#[test]
fn gemm_mma_q8_r256_matches_dequantized_reference() {
    let Some((cuda, k)) = gpu() else { return };
    if !k.has_mma() {
        eprintln!("SKIP: device below sm_75 — no tensor-core module");
        return;
    }
    for (out_dim, in_dim, ntok) in [
        (16usize, 64usize, 5usize),
        (16, 64, 128),
        (16, 64, 200),
        (16, 64, 256),
    ] {
        gemm_mma_case(&cuda, &k, out_dim, in_dim, ntok, true);
    }
    // Real shapes at the token count a 220-token prompt would actually use in
    // one r256 call — the whole reason the kernel exists. Currently
    // unreachable: this test fails at (16, 64, 5) first. Kept so that whoever
    // fixes the indexing defect is held to production shapes, not to the toy
    // case that reported 41-43% on a wrong answer.
    for (out_dim, in_dim, _) in REAL_SHAPES {
        gemm_mma_case(&cuda, &k, out_dim, in_dim, 220, true);
    }
}

fn gemm_mma_case(
    cuda: &Cuda,
    k: &KernelSet,
    out_dim: usize,
    in_dim: usize,
    ntok: usize,
    r256: bool,
) {
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

    // Sized per allocation rather than by a rule of thumb. The old
    // `(ntok_pad * in_dim) * 5 + 8192` happened to cover the toy shapes: the
    // *5 is x-as-f32 (4) plus x-as-int8 (1), leaving the per-32 scale buffer
    // to come out of the 8 KiB slack. At in_dim=4864 that buffer alone is
    // 38 KiB and the allocation would have failed — a limit invisible while
    // every case was 16x64.
    let bytes = (qs.len()                        // weights, int8
        + scales.len()                           // weight scales, f16
        + ntok_pad * in_dim * 4                  // x, f32
        + ntok_pad * in_dim                      // x, quantized int8
        + (ntok_pad * in_dim / 32) * 4           // x scales, f32
        + ntok * out_dim * 4                     // y
        + 64 * 1024) as u64; // alignment slack, six allocations
    let mut buf = BackendBuffer::new(cuda, bytes).unwrap();
    let dwqs = buf.alloc(qs.len() as u64).unwrap().dptr;
    cuda.htod(dwqs, &qs).unwrap();
    let dwsc = buf.alloc(scales.len() as u64).unwrap().dptr;
    cuda.htod(dwsc, &scales).unwrap();
    let dx = upload(cuda, &mut buf, &x);
    // Quantize all padded rows in one pass, exactly as prefill does.
    let d_qs = buf.alloc((ntok_pad * in_dim) as u64).unwrap().dptr;
    let d_scales = buf.alloc_f32(ntok_pad * in_dim / 32).unwrap().dptr;
    k.quantize_q8(cuda, dx, d_qs, d_scales, (ntok_pad * in_dim) as u32)
        .unwrap();
    let dy = buf.alloc_f32(ntok * out_dim).unwrap().dptr;
    if r256 {
        k.gemm_mma_q8_r256(
            cuda,
            dwqs,
            dwsc,
            d_qs,
            d_scales,
            dy,
            out_dim as u32,
            in_dim as u32,
            ntok as u32,
        )
        .unwrap();
    } else {
        k.gemm_mma_q8(
            cuda,
            dwqs,
            dwsc,
            d_qs,
            d_scales,
            dy,
            out_dim as u32,
            in_dim as u32,
            ntok as u32,
        )
        .unwrap();
    }
    cuda.synchronize().unwrap();
    let mut got = vec![0f32; ntok * out_dim];
    cuda.dtoh_f32(&mut got, dy).unwrap();
    buf.free(cuda).unwrap();

    let name = if r256 {
        "gemm_mma_q8_r256"
    } else {
        "gemm_mma_q8"
    };
    // The shape belongs in the label: with several cases per test, a message
    // that names only ntok forces you to reproduce the failure before you can
    // tell which matrix produced it.
    assert_close(
        &got,
        &want,
        EPS_Q8_GEMV,
        &format!(
            "{name}(out={out_dim}, in={in_dim}, ntok={ntok}, blocks={})",
            out_dim.div_ceil(64)
        ),
    );
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
fn wave11_fused_rms_q8_is_bit_exact_to_unfused_chains() {
    let Some((cuda, k)) = gpu() else { return };
    let (rows, dim) = (3usize, 896usize);
    let n = rows * dim;
    let eps = 1e-5f32;
    let mut x = randv(n, 0x1101, 3.0);
    let mut residual = randv(n, 0x1102, 0.5);
    let w = randv(dim, 0x1103, 1.0);
    // Regression cases: tied +/-amax followed by an all-zero K32 block.
    x[0] = 4.0;
    x[1] = -4.0;
    x[32..64].fill(0.0);
    residual[32..64].fill(0.0);

    for add_residual in [false, true] {
        let bytes = (n * 5 * 4 + n * 2 + (n / 32) * 2 * 4 + 16_384) as u64;
        let mut buf = BackendBuffer::new(&cuda, bytes).unwrap();
        let dx_legacy = upload(&cuda, &mut buf, &x);
        let dx_fused = upload(&cuda, &mut buf, &x);
        let dres = upload(&cuda, &mut buf, &residual);
        let dw = upload(&cuda, &mut buf, &w);
        let dout_legacy = buf.alloc_f32(n).unwrap().dptr;
        let dout_fused = buf.alloc_f32(n).unwrap().dptr;
        let dqs_legacy = buf.alloc(n as u64).unwrap().dptr;
        let dqs_fused = buf.alloc(n as u64).unwrap().dptr;
        let dsc_legacy = buf.alloc_f32(n / 32).unwrap().dptr;
        let dsc_fused = buf.alloc_f32(n / 32).unwrap().dptr;

        if add_residual {
            k.add(&cuda, dx_legacy, dres, n as u32).unwrap();
        }
        k.rms_norm_rows(
            &cuda,
            dx_legacy,
            dw,
            dout_legacy,
            dim as u32,
            eps,
            rows as u32,
        )
        .unwrap();
        k.quantize_q8(&cuda, dout_legacy, dqs_legacy, dsc_legacy, n as u32)
            .unwrap();
        k.rms_quantize_q8_rows(
            &cuda,
            dx_fused,
            add_residual.then_some(dres),
            dw,
            dout_fused,
            dqs_fused,
            dsc_fused,
            dim as u32,
            eps,
            rows as u32,
        )
        .unwrap();
        cuda.synchronize().unwrap();

        let mut out_legacy = vec![0f32; n];
        let mut out_fused = vec![0f32; n];
        let mut sc_legacy = vec![0f32; n / 32];
        let mut sc_fused = vec![0f32; n / 32];
        cuda.dtoh_f32(&mut out_legacy, dout_legacy).unwrap();
        cuda.dtoh_f32(&mut out_fused, dout_fused).unwrap();
        cuda.dtoh_f32(&mut sc_legacy, dsc_legacy).unwrap();
        cuda.dtoh_f32(&mut sc_fused, dsc_fused).unwrap();
        assert_bits_eq(&out_fused, &out_legacy, "Wave11 RMS output");
        assert_bits_eq(&sc_fused, &sc_legacy, "Wave11 RMS Q8 scales");
        assert_eq!(
            download_bytes(&cuda, dqs_fused, n),
            download_bytes(&cuda, dqs_legacy, n),
            "Wave11 RMS Q8 bytes"
        );
        if add_residual {
            let mut x_legacy = vec![0f32; n];
            let mut x_fused = vec![0f32; n];
            cuda.dtoh_f32(&mut x_legacy, dx_legacy).unwrap();
            cuda.dtoh_f32(&mut x_fused, dx_fused).unwrap();
            assert_bits_eq(&x_fused, &x_legacy, "Wave11 fused residual");
        }
        buf.free(&cuda).unwrap();
    }
}

#[test]
fn wave11_fused_silu_q8_is_bit_exact_to_unfused_chain() {
    let Some((cuda, k)) = gpu() else { return };
    let n = 3usize * 4864;
    let mut gate = randv(n, 0x1111, 4.0);
    let up = randv(n, 0x1112, 4.0);
    gate[0] = 5.0;
    gate[1] = -5.0;
    gate[32..64].fill(0.0);
    let bytes = (n * 5 * 4 + n * 2 + (n / 32) * 2 * 4 + 16_384) as u64;
    let mut buf = BackendBuffer::new(&cuda, bytes).unwrap();
    let dgate_legacy = upload(&cuda, &mut buf, &gate);
    let dgate_fused = upload(&cuda, &mut buf, &gate);
    let dup = upload(&cuda, &mut buf, &up);
    let dqs_legacy = buf.alloc(n as u64).unwrap().dptr;
    let dqs_fused = buf.alloc(n as u64).unwrap().dptr;
    let dsc_legacy = buf.alloc_f32(n / 32).unwrap().dptr;
    let dsc_fused = buf.alloc_f32(n / 32).unwrap().dptr;

    k.silu_mul(&cuda, dgate_legacy, dup, n as u32).unwrap();
    k.quantize_q8(&cuda, dgate_legacy, dqs_legacy, dsc_legacy, n as u32)
        .unwrap();
    k.silu_mul_quantize_q8(&cuda, dgate_fused, dup, dqs_fused, dsc_fused, n as u32)
        .unwrap();
    cuda.synchronize().unwrap();

    let mut gate_legacy = vec![0f32; n];
    let mut gate_fused = vec![0f32; n];
    let mut sc_legacy = vec![0f32; n / 32];
    let mut sc_fused = vec![0f32; n / 32];
    cuda.dtoh_f32(&mut gate_legacy, dgate_legacy).unwrap();
    cuda.dtoh_f32(&mut gate_fused, dgate_fused).unwrap();
    cuda.dtoh_f32(&mut sc_legacy, dsc_legacy).unwrap();
    cuda.dtoh_f32(&mut sc_fused, dsc_fused).unwrap();
    assert_bits_eq(&gate_fused, &gate_legacy, "Wave11 SwiGLU output");
    assert_bits_eq(&sc_fused, &sc_legacy, "Wave11 SwiGLU Q8 scales");
    assert_eq!(
        download_bytes(&cuda, dqs_fused, n),
        download_bytes(&cuda, dqs_legacy, n),
        "Wave11 SwiGLU Q8 bytes"
    );
    buf.free(&cuda).unwrap();
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
        assert!(
            (total - 1.0).abs() < 1e-5,
            "softmax must sum to 1, got {total}"
        );
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
            let (a, b) = if neox {
                (i, i + half)
            } else {
                (2 * i, 2 * i + 1)
            };
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
        let mut buf = BackendBuffer::new(&cuda, ((x.len() + head_dim) * 4 + 4096) as u64).unwrap();
        let dx = upload(&cuda, &mut buf, &x);
        let dcos = upload(&cuda, &mut buf, &cos);
        let dsin = upload(&cuda, &mut buf, &sin);
        let dpos = buf.alloc(4).unwrap().dptr;
        cuda.htod(dpos, &0u32.to_ne_bytes()).unwrap();
        k.rope(
            &cuda,
            dx,
            dcos,
            dsin,
            n_heads as u32,
            head_dim as u32,
            neox,
            dpos,
        )
        .unwrap();
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
    k.gemv(&cuda, dk, dq, dscores, cached_len as u32, head_dim as u32)
        .unwrap();
    k.softmax_scale(&cuda, dscores, cached_len as u32, scale)
        .unwrap();
    k.gemv_t(&cuda, dv, dscores, dout, cached_len as u32, head_dim as u32)
        .unwrap();
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
    let (head_dim, n_heads, n_kv, cached_len, max_ctx) =
        (64usize, 8usize, 2usize, 100usize, 128usize);
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
    cuda.htod(dclen, &(cached_len as u32).to_ne_bytes())
        .unwrap();

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

/// M2.3 Stage 1b: the batched-over-tokens attention, reached through
/// `attention::prefill`, must equal the per-token reference for EVERY row, with
/// row t attending to exactly its own causal prefix
/// (cached_len = pos_seq[base+t] + 1) — the kernel that lets one launch replace
/// prefill's serial per-token loop.
///
/// This asserts the path as well as the numbers, and Wave 15A moved it: the
/// default is now `Qk4` rather than `Rows`. That is the assertion doing its job
/// — a silent change of default would be exactly the kind of thing worth
/// failing a build over. What it guards is unchanged and is the point of a
/// retention: **an engine with nothing configured produces reference-correct
/// output**. The row kernel itself keeps its own coverage in
/// `wave15_four_chain_qk_is_bit_exact_to_the_retained_attention`, which
/// launches it directly and compares bit for bit.
#[test]
fn attn_decode_rows_matches_per_token_reference() {
    let Some((cuda, k)) = gpu() else { return };
    let (head_dim, n_heads, n_kv, max_ctx) = (64usize, 4usize, 2usize, 64usize);
    let (base, ntok) = (5usize, 3usize); // chunk starts mid-sequence
    let heads_per_kv = n_heads / n_kv;
    let head_stride = max_ctx * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // KV cache with rows 0..base+ntok populated per KV head.
    let filled = base + ntok;
    let mut kc = vec![0f32; n_kv * head_stride];
    let mut vc = vec![0f32; n_kv * head_stride];
    for kvh in 0..n_kv {
        let sk = randv(filled * head_dim, 80 + kvh as u64, 1.0);
        let sv = randv(filled * head_dim, 85 + kvh as u64, 1.0);
        kc[kvh * head_stride..kvh * head_stride + sk.len()].copy_from_slice(&sk);
        vc[kvh * head_stride..kvh * head_stride + sv.len()].copy_from_slice(&sv);
    }
    // Queries for every token row.
    let q = randv(ntok * n_heads * head_dim, 90, 1.0);

    // Reference: row t, head h attends to the first base+t+1 rows only.
    let mut want = vec![0f32; ntok * n_heads * head_dim];
    for t in 0..ntok {
        for h in 0..n_heads {
            let kvh = h / heads_per_kv;
            let len = (base + t + 1) * head_dim;
            let krows = &kc[kvh * head_stride..kvh * head_stride + len];
            let vrows = &vc[kvh * head_stride..kvh * head_stride + len];
            let qoff = t * n_heads * head_dim + h * head_dim;
            let out = attention_ref(&q[qoff..qoff + head_dim], krows, vrows, head_dim);
            want[qoff..qoff + head_dim].copy_from_slice(&out);
        }
    }

    let bytes = ((q.len() + kc.len() + vc.len() + q.len()) * 4 + 8192) as u64;
    let mut buf = BackendBuffer::new(&cuda, bytes).unwrap();
    let dq = upload(&cuda, &mut buf, &q);
    let dk = upload(&cuda, &mut buf, &kc);
    let dv = upload(&cuda, &mut buf, &vc);
    let dout = buf.alloc_f32(q.len()).unwrap().dptr;
    // pos_seq identity array, as the model uploads at load.
    let pos_vals: Vec<u32> = (0..=(max_ctx as u32)).collect();
    let dpos = buf.alloc(((max_ctx + 1) * 4) as u64).unwrap().dptr;
    let pos_bytes =
        unsafe { std::slice::from_raw_parts(pos_vals.as_ptr().cast::<u8>(), pos_vals.len() * 4) };
    cuda.htod(dpos, pos_bytes).unwrap();

    // Through the production entry point, so this test covers the dispatch
    // as well as the kernel. 2 heads per KV head is not the GQA7 shape, so
    // the retained row path must be what ran; asserting it stops the test
    // from silently grading a different kernel than its name claims.
    let call = glcuda::attention::VLAttentionCall {
        n_tokens: ntok as u32,
        pos_base: base as u32,
        n_heads: n_heads as u32,
        n_kv_heads: n_kv as u32,
        head_dim: head_dim as u32,
        head_stride: head_stride as u32,
        scale,
    };
    assert_eq!(call.score_capacity(), filled as u32);
    let path = glcuda::attention::prefill(
        &cuda,
        &k,
        dq,
        (n_heads * head_dim) as u32,
        dk,
        dv,
        dout,
        dpos + (base * 4) as u64,
        &call,
    )
    .unwrap();
    assert_eq!(path, glcuda::attention::ENAttentionPath::Qk4);
    cuda.synchronize().unwrap();

    let mut got = vec![0f32; q.len()];
    cuda.dtoh_f32(&mut got, dout).unwrap();
    buf.free(&cuda).unwrap();

    assert_close(&got, &want, EPS_MATMUL, "attn_decode_rows");
}

#[test]
fn wave11_gqa7_is_bit_exact_to_retained_attention_at_prompt_tail() {
    let Some((cuda, k)) = gpu() else { return };
    let (head_dim, n_heads, n_kv, max_ctx) = (64usize, 14usize, 2usize, 288usize);
    let (base, ntok) = (236usize, 8usize);
    let filled = base + ntok;
    let heads_per_kv = 7usize;
    let head_stride = max_ctx * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut kc = vec![0f32; n_kv * head_stride];
    let mut vc = vec![0f32; n_kv * head_stride];
    for kvh in 0..n_kv {
        let sk = randv(filled * head_dim, 0x1120 + kvh as u64, 1.0);
        let sv = randv(filled * head_dim, 0x1130 + kvh as u64, 1.0);
        kc[kvh * head_stride..kvh * head_stride + sk.len()].copy_from_slice(&sk);
        vc[kvh * head_stride..kvh * head_stride + sv.len()].copy_from_slice(&sv);
    }
    let q = randv(ntok * n_heads * head_dim, 0x1140, 1.0);
    let bytes = ((q.len() * 3 + kc.len() + vc.len()) * 4 + (max_ctx + 1) * 4 + 32_768) as u64;
    let mut buf = BackendBuffer::new(&cuda, bytes).unwrap();
    let dq = upload(&cuda, &mut buf, &q);
    let dk = upload(&cuda, &mut buf, &kc);
    let dv = upload(&cuda, &mut buf, &vc);
    let dout_legacy = buf.alloc_f32(q.len()).unwrap().dptr;
    let dout_gqa7 = buf.alloc_f32(q.len()).unwrap().dptr;
    let pos_vals: Vec<u32> = (0..=(max_ctx as u32)).collect();
    let dpos = buf.alloc(((max_ctx + 1) * 4) as u64).unwrap().dptr;
    let pos_bytes =
        unsafe { std::slice::from_raw_parts(pos_vals.as_ptr().cast::<u8>(), pos_vals.len() * 4) };
    cuda.htod(dpos, pos_bytes).unwrap();
    let pos_base = dpos + (base * 4) as u64;

    k.attn_decode_rows_legacy(
        &cuda,
        dq,
        dk,
        dv,
        dout_legacy,
        n_heads as u32,
        head_dim as u32,
        pos_base,
        heads_per_kv as u32,
        head_stride as u32,
        scale,
        ntok as u32,
        filled as u32,
        (n_heads * head_dim) as u32,
    )
    .unwrap();
    k.attn_decode_rows_gqa7(
        &cuda,
        dq,
        dk,
        dv,
        dout_gqa7,
        n_heads as u32,
        head_dim as u32,
        pos_base,
        heads_per_kv as u32,
        head_stride as u32,
        scale,
        ntok as u32,
        filled as u32,
        (n_heads * head_dim) as u32,
    )
    .unwrap();
    cuda.synchronize().unwrap();

    let mut legacy = vec![0f32; q.len()];
    let mut gqa7 = vec![0f32; q.len()];
    cuda.dtoh_f32(&mut legacy, dout_legacy).unwrap();
    cuda.dtoh_f32(&mut gqa7, dout_gqa7).unwrap();
    buf.free(&cuda).unwrap();
    assert_bits_eq(&gqa7, &legacy, "Wave11 GQA7 attention");
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

    let mut buf =
        BackendBuffer::new(&cuda, ((n_kv * head_stride + src.len()) * 4 + 4096) as u64).unwrap();
    let dst = buf.alloc_f32(n_kv * head_stride).unwrap().dptr; // zeroed region
    cuda.htod_f32(dst, &vec![0f32; n_kv * head_stride]).unwrap();
    let dsrc = upload(&cuda, &mut buf, &src);
    let dpos = buf.alloc(4).unwrap().dptr;
    cuda.htod(dpos, &(pos as u32).to_ne_bytes()).unwrap();

    k.kv_write(
        &cuda,
        dst,
        dsrc,
        dpos,
        head_dim as u32,
        n_kv as u32,
        head_stride as u32,
    )
    .unwrap();
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
        assert_eq!(
            *g,
            b + 6.0,
            "graph replay did not execute the recorded adds"
        );
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
