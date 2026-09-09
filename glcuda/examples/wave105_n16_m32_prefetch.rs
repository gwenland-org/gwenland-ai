//! Wave 105 direct T4 gate for the N16/M32 register-prefetch candidate.
//!
//! This executable produces direct-kernel feasibility evidence only. It does
//! not establish production throughput or change the retained dispatch.

use std::time::Instant;

use glcuda::buffer::BackendBuffer;
use glcuda::driver::{cuda_available, Cuda};
use glcuda::kernels::KernelSet;
use glcuda::repack::q8_0_soa_to_bstage;

const OUT_DIM: usize = 896;
const IN_DIM: usize = 4_864;
const NTOK: usize = 244;
const WARMUP: usize = 10;
const ITERS: usize = 100;
const REPEATS: usize = 5;
const DIRECT_GATE: f64 = 1.10;

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

fn q8_weights() -> (Vec<u8>, Vec<u8>) {
    let qs = (0..OUT_DIM * IN_DIM)
        .map(|i| ((i.wrapping_mul(29).wrapping_add(i / 97)) % 255) as u8)
        .collect();
    // Exact f16 2^-7 (0x2000), little-endian like the retained SoA image.
    let scales = (0..OUT_DIM * (IN_DIM / 32))
        .flat_map(|_| [0x00, 0x20])
        .collect();
    (qs, scales)
}

fn activations(rows: usize) -> Vec<f32> {
    (0..rows * IN_DIM)
        .map(|i| ((i.wrapping_mul(37) % 251) as f32 - 125.0) / 127.0)
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        println!("[wave105-n16-m32-prefetch] no CUDA device; nothing measured");
        return Ok(());
    }
    let cuda = Cuda::probe()?;
    if (cuda.info.sm_major, cuda.info.sm_minor) != (7, 5) {
        return Err("Wave 105 requires an SM75 device".into());
    }
    let kernels = KernelSet::load(&cuda)?;
    if !kernels.gemm_n16_m32_prefetch_enabled(OUT_DIM as u32, IN_DIM as u32, NTOK as u32) {
        return Err("set GLCUDA_GRID2D=1 GLCUDA_NTILE128=1 GLCUDA_BSTAGE=1 \
             GLCUDA_GEMM_N16=1 GLCUDA_GEMM_N16_M32_PREFETCH=1"
            .into());
    }
    if !kernels.gemm_n16_uses_m32(OUT_DIM as u32, NTOK as u32) {
        return Err("Wave 105 production shape did not select M32".into());
    }

    let retained_resource = kernels
        .wave27_n16_m32_resource_kernel()
        .ok_or("retained N16/M32 resource kernel unavailable")?;
    let candidate_resource = kernels
        .wave105_n16_m32_prefetch_resource_kernel()
        .ok_or("Wave 105 resource kernel unavailable")?;
    let retained_active = cuda
        .max_active_blocks_per_sm(retained_resource, 256, 0)
        .ok_or("retained occupancy query unavailable")?;
    let candidate_active = cuda
        .max_active_blocks_per_sm(candidate_resource, 256, 0)
        .ok_or("candidate occupancy query unavailable")?;
    if retained_active < 3 || candidate_active < 3 {
        return Err(format!(
            "occupancy gate failed: retained={retained_active}, candidate={candidate_active}"
        )
        .into());
    }

    let ntok_pad = NTOK.div_ceil(8) * 8;
    let nb = IN_DIM / 32;
    let (qs, scales) = q8_weights();
    let (tiled_qs, tiled_scales) = q8_0_soa_to_bstage(&qs, &scales, OUT_DIM, IN_DIM)?;
    let x = activations(ntok_pad);
    let bytes = (tiled_qs.len()
        + tiled_scales.len()
        + x.len() * 4
        + ntok_pad * IN_DIM
        + ntok_pad * nb * 4
        + 2 * NTOK * OUT_DIM * 4
        + 2 * 1_048_576) as u64;
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
    let retained_out = buf.alloc_f32(NTOK * OUT_DIM)?.dptr;
    let candidate_out = buf.alloc_f32(NTOK * OUT_DIM)?.dptr;

    let retained = |out| {
        kernels.gemm_mma_q8_bstage_n16(
            &cuda,
            wqs,
            wsc,
            xqs,
            xsc,
            out,
            OUT_DIM as u32,
            IN_DIM as u32,
            NTOK as u32,
        )
    };
    let candidate = |out| {
        kernels.gemm_mma_q8_bstage_n16_m32_prefetch(
            &cuda,
            wqs,
            wsc,
            xqs,
            xsc,
            out,
            OUT_DIM as u32,
            IN_DIM as u32,
            NTOK as u32,
        )
    };

    retained(retained_out)?;
    candidate(candidate_out)?;
    cuda.synchronize()?;
    let mut host_retained = vec![0f32; NTOK * OUT_DIM];
    let mut host_candidate = vec![0f32; NTOK * OUT_DIM];
    cuda.dtoh_f32(&mut host_retained, retained_out)?;
    cuda.dtoh_f32(&mut host_candidate, candidate_out)?;
    let first_mismatch = host_retained
        .iter()
        .zip(&host_candidate)
        .position(|(a, b)| a.to_bits() != b.to_bits())
        .map_or(-1, |i| i as i64);
    if first_mismatch >= 0 {
        return Err(format!("bit-exact gate failed at output {first_mismatch}").into());
    }

    // Counterbalanced within one invocation to suppress T4 clock drift.
    let (mut retained_us, mut candidate_us) = (0.0, 0.0);
    for _ in 0..REPEATS {
        retained_us += timed(&cuda, || retained(retained_out))?;
        candidate_us += timed(&cuda, || candidate(candidate_out))?;
        candidate_us += timed(&cuda, || candidate(candidate_out))?;
        retained_us += timed(&cuda, || retained(retained_out))?;
    }
    let observations = (2 * REPEATS) as f64;
    retained_us /= observations;
    candidate_us /= observations;
    let speedup = retained_us / candidate_us;

    println!(
        "[wave105-resource] {{\"retained_kernel\":\"gl_gemm_mma_q8_bstage_n16_m32\",\
\"candidate_kernel\":\"gl_gemm_mma_q8_bstage_n16_m32_prefetch\",\
\"threads\":256,\"dynamic_shared_bytes\":0,\
\"retained_active_blocks_per_sm\":{},\"candidate_active_blocks_per_sm\":{}}}",
        retained_active, candidate_active
    );
    println!(
        "[wave105-direct] {{\"warmup\":{},\"iters\":{},\"repeats\":{},\
\"out_dim\":{},\"in_dim\":{},\"ntok\":{},\
\"retained_us\":{:.3},\"candidate_us\":{:.3},\"speedup\":{:.4},\
\"bit_exact\":true,\"first_mismatch\":-1,\"gate\":{:.2},\"pass\":{}}}",
        WARMUP,
        ITERS,
        REPEATS,
        OUT_DIM,
        IN_DIM,
        NTOK,
        retained_us,
        candidate_us,
        speedup,
        DIRECT_GATE,
        speedup >= DIRECT_GATE,
    );
    buf.free(&cuda)?;
    if speedup < DIRECT_GATE {
        return Err(format!("direct speed gate failed: {speedup:.4}x < {DIRECT_GATE:.2}x").into());
    }
    Ok(())
}
