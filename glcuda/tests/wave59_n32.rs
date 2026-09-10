//! Wave 59 device parity gate. Run serially because it owns process flags:
//! `cargo test -p glcuda --test wave59_n32 -- --test-threads=1`.

use glcuda::buffer::BackendBuffer;
use glcuda::driver::{cuda_available, Cuda};
use glcuda::kernels::KernelSet;
use glcuda::repack::q8_0_soa_to_bstage;

#[test]
fn n32_m32_is_bit_exact_to_retained_n16_m32() {
    if !cuda_available() {
        eprintln!("SKIP: no CUDA driver/device on this machine");
        return;
    }
    for flag in [
        "GLCUDA_GRID2D",
        "GLCUDA_NTILE128",
        "GLCUDA_BSTAGE",
        "GLCUDA_GEMM_N16",
        "GLCUDA_GEMM_N32",
    ] {
        std::env::set_var(flag, "1");
    }
    let cuda = Cuda::probe().expect("CUDA probe");
    if (cuda.info.sm_major, cuda.info.sm_minor) < (7, 5) {
        eprintln!("SKIP: Wave 59 requires sm_75 or newer");
        return;
    }
    let kernels = KernelSet::load(&cuda).expect("Wave 59 PTX must JIT");
    assert!(kernels.gemm_n32_enabled());

    for (out_dim, in_dim, ntok) in [(136usize, 160usize, 17usize), (896, 256, 33)] {
        exact_case(&cuda, &kernels, out_dim, in_dim, ntok);
    }
}

fn exact_case(cuda: &Cuda, kernels: &KernelSet, out_dim: usize, in_dim: usize, ntok: usize) {
    let ntok_pad = ntok.div_ceil(8) * 8;
    let nb = in_dim / 32;
    let qs: Vec<u8> = (0..out_dim * in_dim)
        .map(|i| ((i.wrapping_mul(29).wrapping_add(i / 97)) % 255) as u8)
        .collect();
    let scales: Vec<u8> = (0..out_dim * nb).flat_map(|_| [0x00, 0x20]).collect();
    let (tiled_qs, tiled_scales) =
        q8_0_soa_to_bstage(&qs, &scales, out_dim, in_dim).expect("B-stage repack");
    let x: Vec<f32> = (0..ntok_pad * in_dim)
        .map(|i| ((i.wrapping_mul(37) % 251) as f32 - 125.0) / 127.0)
        .collect();
    let bytes = (tiled_qs.len()
        + tiled_scales.len()
        + x.len() * 4
        + ntok_pad * in_dim
        + ntok_pad * nb * 4
        + 2 * ntok * out_dim * 4
        + 64 * 1024) as u64;
    let mut buf = BackendBuffer::new(cuda, bytes).expect("device buffer");
    let wqs = buf.alloc(tiled_qs.len() as u64).unwrap().dptr;
    let wsc = buf.alloc(tiled_scales.len() as u64).unwrap().dptr;
    cuda.htod(wqs, &tiled_qs).unwrap();
    cuda.htod(wsc, &tiled_scales).unwrap();
    let xf = buf.alloc_f32(x.len()).unwrap().dptr;
    cuda.htod_f32(xf, &x).unwrap();
    let xqs = buf.alloc((ntok_pad * in_dim) as u64).unwrap().dptr;
    let xsc = buf.alloc_f32(ntok_pad * nb).unwrap().dptr;
    kernels
        .quantize_q8(cuda, xf, xqs, xsc, (ntok_pad * in_dim) as u32)
        .unwrap();
    let retained = buf.alloc_f32(ntok * out_dim).unwrap().dptr;
    let candidate = buf.alloc_f32(ntok * out_dim).unwrap().dptr;

    kernels
        .gemm_mma_q8_bstage_n16(
            cuda,
            wqs,
            wsc,
            xqs,
            xsc,
            retained,
            out_dim as u32,
            in_dim as u32,
            ntok as u32,
        )
        .unwrap();
    kernels
        .gemm_mma_q8_bstage_n32_m32(
            cuda,
            wqs,
            wsc,
            xqs,
            xsc,
            candidate,
            out_dim as u32,
            in_dim as u32,
            ntok as u32,
        )
        .unwrap();
    cuda.synchronize().unwrap();

    let mut a = vec![0f32; ntok * out_dim];
    let mut b = vec![0f32; ntok * out_dim];
    cuda.dtoh_f32(&mut a, retained).unwrap();
    cuda.dtoh_f32(&mut b, candidate).unwrap();
    buf.free(cuda).unwrap();
    let mismatch = a
        .iter()
        .zip(&b)
        .position(|(left, right)| left.to_bits() != right.to_bits());
    assert_eq!(
        mismatch, None,
        "Wave 59 mismatch at {mismatch:?}: out={out_dim} in={in_dim} ntok={ntok}"
    );
}
