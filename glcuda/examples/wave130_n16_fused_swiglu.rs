//! Wave 130 direct T4 gate for N16-prefetch gate/up -> SwiGLU -> Q8_0.
//!
//! The retained and candidate arms share one context and allocation.
//! Correctness is byte/bit exact before diagnostic timing is reported.

use std::time::Instant;

use glcuda::buffer::BackendBuffer;
use glcuda::driver::{cuda_available, Cuda};
use glcuda::kernels::KernelSet;
use glcuda::repack::q8_0_soa_to_bstage;

const HIDDEN: usize = 4_864;
const OUT_DIM: usize = 2 * HIDDEN;
const IN_DIM: usize = 896;
const NTOK: usize = 244;
const WARMUP: usize = 10;
const ITERS: usize = 100;
const REPEATS: usize = 5;

fn timed<F>(cuda: &Cuda, mut launch: F) -> Result<f64, glcore::GlError>
where
    F: FnMut() -> Result<(), glcore::GlError>,
{
    for _ in 0..WARMUP {
        launch()?;
    }
    cuda.synchronize()?;
    let start = Instant::now();
    for _ in 0..ITERS {
        launch()?;
    }
    cuda.synchronize()?;
    Ok(start.elapsed().as_secs_f64() * 1e6 / ITERS as f64)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        println!("[wave130-direct] no CUDA device; nothing measured");
        return Ok(());
    }
    let cuda = Cuda::probe()?;
    if (cuda.info.sm_major, cuda.info.sm_minor) != (7, 5) {
        return Err("Wave 130 requires an SM75 device".into());
    }
    let kernels = KernelSet::load(&cuda)?;
    if !kernels.n16_fused_swiglu_enabled(HIDDEN as u32, IN_DIM as u32, NTOK as u32) {
        return Err(
            "set the retained Wave 123/Wave 88 flags plus GLCUDA_N16_FUSED_SWIGLU=1".into(),
        );
    }
    let resource = kernels
        .wave129_fused_swiglu_resource_kernel()
        .ok_or("Wave 130 resource kernel unavailable")?;
    let active = cuda
        .max_active_blocks_per_sm(resource, 256, 0)
        .ok_or("Wave 130 occupancy query unavailable")?;
    if active < 3 {
        return Err(format!("Wave 130 occupancy gate failed: {active} blocks/SM < 3").into());
    }

    let nb = IN_DIM / 32;
    let ntok_pad = NTOK.div_ceil(8) * 8;
    let qs: Vec<u8> = (0..OUT_DIM * IN_DIM)
        .map(|i| ((i.wrapping_mul(29).wrapping_add(i / 97)) % 255) as u8)
        .collect();
    let scales: Vec<u8> = (0..OUT_DIM * nb)
        .flat_map(|i| {
            let bits = 0x1800u16 + ((i % 5) as u16) * 0x0400;
            bits.to_le_bytes()
        })
        .collect();
    let (tiled_qs, tiled_scales) = q8_0_soa_to_bstage(&qs, &scales, OUT_DIM, IN_DIM)?;
    let x: Vec<f32> = (0..ntok_pad * IN_DIM)
        .map(|i| ((i.wrapping_mul(37) % 251) as f32 - 125.0) / 127.0)
        .collect();
    let q_bytes = NTOK * HIDDEN;
    let scale_count = q_bytes / 32;
    let bytes = (tiled_qs.len()
        + tiled_scales.len()
        + x.len() * 4
        + ntok_pad * IN_DIM
        + ntok_pad * nb * 4
        + NTOK * OUT_DIM * 4
        + 2 * q_bytes
        + 2 * scale_count * 4
        + 4 * 1_048_576) as u64;
    let mut buf = BackendBuffer::new(&cuda, bytes)?;
    let wqs = buf.alloc(tiled_qs.len() as u64)?.dptr;
    let wsc = buf.alloc(tiled_scales.len() as u64)?.dptr;
    cuda.htod(wqs, &tiled_qs)?;
    cuda.htod(wsc, &tiled_scales)?;
    let xf = buf.alloc_f32(x.len())?.dptr;
    cuda.htod_f32(xf, &x)?;
    let xqs = buf.alloc((ntok_pad * IN_DIM) as u64)?.dptr;
    let xsc = buf.alloc_f32(ntok_pad * nb)?.dptr;
    kernels.quantize_q8(&cuda, xf, xqs, xsc, (ntok_pad * IN_DIM) as u32)?;
    let gate_up = buf.alloc_f32(NTOK * OUT_DIM)?.dptr;
    let retained_qs = buf.alloc(q_bytes as u64)?.dptr;
    let retained_sc = buf.alloc_f32(scale_count)?.dptr;
    let candidate_qs = buf.alloc(q_bytes as u64)?.dptr;
    let candidate_sc = buf.alloc_f32(scale_count)?.dptr;
    let up_tile = HIDDEN / 128;
    let up_qoff = (up_tile * nb * 128 * 32) as u64;
    let up_soff = (up_tile * nb * 128 * 2) as u64;

    let retained = || {
        kernels.gemm_mma_q8_bstage_n16_prefetch(
            &cuda,
            wqs,
            wsc,
            xqs,
            xsc,
            gate_up,
            OUT_DIM as u32,
            IN_DIM as u32,
            NTOK as u32,
        )?;
        kernels.silu_mul_quantize_q8_stacked_nostore(
            &cuda,
            gate_up,
            retained_qs,
            retained_sc,
            HIDDEN as u32,
            NTOK as u32,
        )
    };
    let candidate = || {
        kernels.gemm_mma_q8_bstage_n16_fused_swiglu(
            &cuda,
            wqs,
            wsc,
            wqs + up_qoff,
            wsc + up_soff,
            xqs,
            xsc,
            candidate_qs,
            candidate_sc,
            HIDDEN as u32,
            IN_DIM as u32,
            NTOK as u32,
        )
    };

    retained()?;
    candidate()?;
    cuda.synchronize()?;
    let mut retained_words = vec![0f32; q_bytes / 4];
    let mut candidate_words = vec![0f32; q_bytes / 4];
    let mut retained_scales = vec![0f32; scale_count];
    let mut candidate_scales = vec![0f32; scale_count];
    cuda.dtoh_f32(&mut retained_words, retained_qs)?;
    cuda.dtoh_f32(&mut candidate_words, candidate_qs)?;
    cuda.dtoh_f32(&mut retained_scales, retained_sc)?;
    cuda.dtoh_f32(&mut candidate_scales, candidate_sc)?;
    let q_mismatch = retained_words
        .iter()
        .zip(&candidate_words)
        .position(|(a, b)| a.to_bits() != b.to_bits());
    let scale_mismatch = retained_scales
        .iter()
        .zip(&candidate_scales)
        .position(|(a, b)| a.to_bits() != b.to_bits());
    if q_mismatch.is_some() || scale_mismatch.is_some() {
        return Err(format!(
            "Wave 130 parity failed: q_word={q_mismatch:?} scale={scale_mismatch:?}"
        )
        .into());
    }

    let (mut retained_us, mut candidate_us) = (0.0, 0.0);
    for _ in 0..REPEATS {
        retained_us += timed(&cuda, retained)?;
        candidate_us += timed(&cuda, candidate)?;
        candidate_us += timed(&cuda, candidate)?;
        retained_us += timed(&cuda, retained)?;
    }
    let observations = (2 * REPEATS) as f64;
    retained_us /= observations;
    candidate_us /= observations;
    println!(
        "[wave130-resource] {{\"kernel\":\"gl_gemm_mma_q8_bstage_n16_fused_swiglu_specialized\",\"threads\":256,\"static_shared_bytes\":16384,\"active_blocks_per_sm\":{active}}}"
    );
    println!(
        "[wave130-direct] {{\"warmup\":{WARMUP},\"iters\":{ITERS},\"repeats\":{REPEATS},\"hidden\":{HIDDEN},\"in_dim\":{IN_DIM},\"ntok\":{NTOK},\"retained_us\":{retained_us:.3},\"candidate_us\":{candidate_us:.3},\"speedup\":{:.4},\"q8_bit_exact\":true,\"scale_bit_exact\":true,\"full_slabs\":3,\"ragged_tail_rows\":52}}",
        retained_us / candidate_us,
    );
    buf.free(&cuda)?;
    Ok(())
}
