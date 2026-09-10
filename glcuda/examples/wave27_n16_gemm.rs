//! Wave 28 direct gate for N16 `ldmatrix` shared-fragment loads.
//!
//! This is a correctness/resource screen, not production retention evidence.

use std::time::Instant;

use glcuda::buffer::BackendBuffer;
use glcuda::driver::{cuda_available, Cuda};
use glcuda::kernels::KernelSet;
use glcuda::repack::q8_0_soa_to_bstage;

const WARMUP: usize = 10;
const ITERS: usize = 100;
const REPEATS: usize = 4;

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

fn q8_weights(out_dim: usize, in_dim: usize) -> (Vec<u8>, Vec<u8>) {
    let qs = (0..out_dim * in_dim)
        .map(|i| ((i.wrapping_mul(29).wrapping_add(i / 97)) % 255) as u8)
        .collect();
    // Exact f16 2^-7 (0x2000), little-endian like the retained SoA image.
    let scales = (0..out_dim * (in_dim / 32))
        .flat_map(|_| [0x00, 0x20])
        .collect();
    (qs, scales)
}

fn activations(rows: usize, in_dim: usize) -> Vec<f32> {
    (0..rows * in_dim)
        .map(|i| ((i.wrapping_mul(37) % 251) as f32 - 125.0) / 127.0)
        .collect()
}

struct Record {
    label: &'static str,
    out_dim: usize,
    in_dim: usize,
    ntok: usize,
    retained_us: f64,
    n16_us: f64,
    bit_exact: bool,
    first_mismatch: i64,
}

impl Record {
    fn json(&self) -> String {
        format!(
            "{{\"shape\":\"{}\",\"out_dim\":{},\"in_dim\":{},\"ntok\":{},\
\"retained_us\":{:.3},\"n16_us\":{:.3},\"ratio\":{:.4},\
\"bit_exact\":{},\"first_mismatch\":{}}}",
            self.label,
            self.out_dim,
            self.in_dim,
            self.ntok,
            self.retained_us,
            self.n16_us,
            self.retained_us / self.n16_us,
            self.bit_exact,
            self.first_mismatch,
        )
    }
}

fn screen(
    cuda: &Cuda,
    kernels: &KernelSet,
    label: &'static str,
    out_dim: usize,
    in_dim: usize,
    ntok: usize,
) -> Result<Record, Box<dyn std::error::Error>> {
    let ntok_pad = ntok.div_ceil(8) * 8;
    let nb = in_dim / 32;
    let (qs, scales) = q8_weights(out_dim, in_dim);
    let (tiled_qs, tiled_scales) = q8_0_soa_to_bstage(&qs, &scales, out_dim, in_dim)?;
    let x = activations(ntok_pad, in_dim);
    let bytes = (tiled_qs.len()
        + tiled_scales.len()
        + x.len() * 4
        + ntok_pad * in_dim
        + ntok_pad * nb * 4
        + 2 * ntok * out_dim * 4
        + 2 * 1_048_576) as u64;
    let mut buf = BackendBuffer::new(cuda, bytes)?;
    let wqs = buf.alloc(tiled_qs.len() as u64)?.dptr;
    let wsc = buf.alloc(tiled_scales.len() as u64)?.dptr;
    cuda.htod(wqs, &tiled_qs)?;
    cuda.htod(wsc, &tiled_scales)?;
    let xf = buf.alloc_f32(x.len())?.dptr;
    cuda.htod_f32(xf, &x)?;
    let xqs = buf.alloc((ntok_pad * in_dim) as u64)?.dptr;
    let xsc = buf.alloc_f32(ntok_pad * nb)?.dptr;
    kernels.quantize_q8(cuda, xf, xqs, xsc, (ntok_pad * in_dim) as u32)?;
    let retained_out = buf.alloc_f32(ntok * out_dim)?.dptr;
    let candidate_out = buf.alloc_f32(ntok * out_dim)?.dptr;

    let retained = |out| {
        kernels.gemm_mma_q8_bstage(
            cuda,
            wqs,
            wsc,
            xqs,
            xsc,
            out,
            out_dim as u32,
            in_dim as u32,
            ntok as u32,
        )
    };
    let candidate = |out| {
        kernels.gemm_mma_q8_bstage_n16(
            cuda,
            wqs,
            wsc,
            xqs,
            xsc,
            out,
            out_dim as u32,
            in_dim as u32,
            ntok as u32,
        )
    };

    retained(retained_out)?;
    candidate(candidate_out)?;
    cuda.synchronize()?;
    let mut host_retained = vec![0f32; ntok * out_dim];
    let mut host_candidate = vec![0f32; ntok * out_dim];
    cuda.dtoh_f32(&mut host_retained, retained_out)?;
    cuda.dtoh_f32(&mut host_candidate, candidate_out)?;
    let first_mismatch = host_retained
        .iter()
        .zip(&host_candidate)
        .position(|(a, b)| a.to_bits() != b.to_bits())
        .map_or(-1, |i| i as i64);

    // Counterbalanced inside one invocation because absolute T4 clocks drift.
    let (mut retained_us, mut n16_us) = (0.0, 0.0);
    for _ in 0..REPEATS {
        retained_us += timed(cuda, || retained(retained_out))?;
        n16_us += timed(cuda, || candidate(candidate_out))?;
        n16_us += timed(cuda, || candidate(candidate_out))?;
        retained_us += timed(cuda, || retained(retained_out))?;
    }
    buf.free(cuda)?;
    let observations = (2 * REPEATS) as f64;
    Ok(Record {
        label,
        out_dim,
        in_dim,
        ntok,
        retained_us: retained_us / observations,
        n16_us: n16_us / observations,
        bit_exact: first_mismatch < 0,
        first_mismatch,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        println!("[wave28-n16] no CUDA device; nothing measured");
        return Ok(());
    }
    let cuda = Cuda::probe()?;
    if (cuda.info.sm_major, cuda.info.sm_minor) < (7, 5) {
        return Err("Wave 28 requires sm_75 or newer".into());
    }
    let kernels = KernelSet::load(&cuda)?;
    if !kernels.gemm_n16_enabled() {
        return Err("set GLCUDA_GRID2D=1 GLCUDA_BSTAGE=1 GLCUDA_GEMM_N16=1".into());
    }
    let wide_resource = kernels
        .wave27_n16_resource_kernel()
        .ok_or("Wave 28 resource kernel unavailable")?;
    let narrow_resource = kernels
        .wave27_n16_m32_resource_kernel()
        .ok_or("Wave 28 M32 resource kernel unavailable")?;
    let wide_active_256 = cuda
        .max_active_blocks_per_sm(wide_resource, 256, 0)
        .ok_or("Wave 27 256-thread occupancy query unavailable")?;
    let wide_active_128 = cuda
        .max_active_blocks_per_sm(wide_resource, 128, 0)
        .ok_or("Wave 27 128-thread occupancy query unavailable")?;
    let narrow_active_256 = cuda
        .max_active_blocks_per_sm(narrow_resource, 256, 0)
        .ok_or("Wave 27 M32 256-thread occupancy query unavailable")?;
    if wide_active_128 < 3 || wide_active_256 < 2 || narrow_active_256 < 2 {
        return Err(format!(
            "Wave 28 occupancy tier failed: wide_active_128={wide_active_128} \
             wide_active_256={wide_active_256} narrow_active_256={narrow_active_256}"
        )
        .into());
    }

    let records = [
        screen(&cuda, &kernels, "ffn_gate_up", 9_728, 896, 244)?,
        screen(&cuda, &kernels, "ffn_down", 896, 4_864, 244)?,
        // Ragged token/output coverage plus a non-production K shape.
        screen(&cuda, &kernels, "ragged_n64_k160", 136, 160, 17)?,
    ];
    let all_exact = records.iter().all(|record| record.bit_exact);
    println!(
        "[wave28-resource] {{\"wide_kernel\":\"gl_gemm_mma_q8_bstage_n16\",\
\"narrow_kernel\":\"gl_gemm_mma_q8_bstage_n16_m32\",\
\"wide_threads_128_active_blocks_per_sm\":{},\
\"wide_threads_256_active_blocks_per_sm\":{},\
\"narrow_threads_256_active_blocks_per_sm\":{},\
\"dynamic_shared_bytes\":0}}",
        wide_active_128, wide_active_256, narrow_active_256
    );
    println!(
        "[wave28-n16] {{\"warmup\":{},\"iters\":{},\"repeats\":{},\
\"records\":[{}],\"all_bit_exact\":{}}}",
        WARMUP,
        ITERS,
        REPEATS,
        records
            .iter()
            .map(Record::json)
            .collect::<Vec<_>>()
            .join(","),
        all_exact,
    );
    if !all_exact {
        return Err("Wave 28 direct bit-exact gate failed".into());
    }
    Ok(())
}
