//! Wave 59 direct T4 screen for the isolated N32/M32 narrow-grid kernel.
//! Correctness and resource checks are hard gates; timing is interleaved
//! diagnostic evidence and does not by itself retain the production path.

use std::time::Instant;

use glcuda::buffer::BackendBuffer;
use glcuda::driver::{cuda_available, Cuda};
use glcuda::kernels::KernelSet;
use glcuda::repack::q8_0_soa_to_bstage;

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

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn q8_weights(out_dim: usize, in_dim: usize) -> (Vec<u8>, Vec<u8>) {
    let qs = (0..out_dim * in_dim)
        .map(|i| ((i.wrapping_mul(29).wrapping_add(i / 97)) % 255) as u8)
        .collect();
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
    candidate_us: f64,
    bit_exact: bool,
    first_mismatch: i64,
}

impl Record {
    fn json(&self) -> String {
        format!(
            "{{\"shape\":\"{}\",\"out_dim\":{},\"in_dim\":{},\"ntok\":{},\
             \"retained_n16_m32_us\":{:.3},\"candidate_n32_m32_us\":{:.3},\
             \"speedup\":{:.4},\"bit_exact\":{},\"first_mismatch\":{}}}",
            self.label,
            self.out_dim,
            self.in_dim,
            self.ntok,
            self.retained_us,
            self.candidate_us,
            self.retained_us / self.candidate_us,
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
    if !kernels.gemm_n16_uses_m32(out_dim as u32, ntok as u32) {
        return Err(format!("{label} did not select the retained N16/M32 control").into());
    }
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
    let candidate = |out| {
        kernels.gemm_mma_q8_bstage_n32_m32(
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

    let (mut retained_samples, mut candidate_samples) = (
        Vec::with_capacity(2 * REPEATS),
        Vec::with_capacity(2 * REPEATS),
    );
    for _ in 0..REPEATS {
        retained_samples.push(timed(cuda, || retained(retained_out))?);
        candidate_samples.push(timed(cuda, || candidate(candidate_out))?);
        candidate_samples.push(timed(cuda, || candidate(candidate_out))?);
        retained_samples.push(timed(cuda, || retained(retained_out))?);
    }
    buf.free(cuda)?;
    Ok(Record {
        label,
        out_dim,
        in_dim,
        ntok,
        retained_us: median(&mut retained_samples),
        candidate_us: median(&mut candidate_samples),
        bit_exact: first_mismatch < 0,
        first_mismatch,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        println!("[wave59-n32] no CUDA device; nothing measured");
        return Ok(());
    }
    let cuda = Cuda::probe()?;
    if (cuda.info.sm_major, cuda.info.sm_minor) != (7, 5) {
        return Err("Wave 59 production gate is pinned to sm_75".into());
    }
    let kernels = KernelSet::load(&cuda)?;
    if !kernels.gemm_n32_enabled() {
        return Err("set GLCUDA_GRID2D=1 GLCUDA_NTILE128=1 GLCUDA_BSTAGE=1 \
             GLCUDA_GEMM_N16=1 GLCUDA_GEMM_N32=1"
            .into());
    }
    let retained_resource = kernels
        .wave27_n16_m32_resource_kernel()
        .ok_or("retained N16/M32 resource kernel unavailable")?;
    let candidate_resource = kernels
        .wave59_n32_m32_resource_kernel()
        .ok_or("Wave 59 resource kernel unavailable")?;
    let retained_active = cuda
        .max_active_blocks_per_sm(retained_resource, 256, 0)
        .ok_or("retained occupancy query unavailable")?;
    let candidate_active = cuda
        .max_active_blocks_per_sm(candidate_resource, 128, 0)
        .ok_or("Wave 59 occupancy query unavailable")?;
    if candidate_active < 6 {
        return Err(format!(
            "Wave 59 occupancy tier failed: candidate_active_blocks_per_sm={candidate_active}"
        )
        .into());
    }

    let records = [
        screen(&cuda, &kernels, "qkv", 1_152, 896, 244)?,
        screen(&cuda, &kernels, "ffn_down", 896, 4_864, 244)?,
        screen(&cuda, &kernels, "ragged", 136, 160, 17)?,
    ];
    let all_exact = records.iter().all(|record| record.bit_exact);
    println!(
        "[wave59-resource] {{\"retained_threads\":256,\"retained_active_blocks_per_sm\":{},\
         \"candidate_threads\":128,\"candidate_active_blocks_per_sm\":{},\
         \"candidate_static_shared_bytes\":8064,\"candidate_maxnreg\":80}}",
        retained_active, candidate_active
    );
    println!(
        "[wave59-n32] {{\"warmup\":{},\"iters\":{},\"repeats\":{},\"records\":[{}],\
         \"all_bit_exact\":{}}}",
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
        return Err("Wave 59 direct bit-exact gate failed".into());
    }
    Ok(())
}
