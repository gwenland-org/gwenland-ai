//! Wave 111 direct gate for cross-layer FFN residual fusion.
//!
//! The retained arm materializes `x += residual` and then launches the fused
//! RMSNorm+Q8 kernel. The candidate asks that same RMSNorm+Q8 kernel to perform
//! the identical f32 add while it reads `x`. This is the exact operation moved
//! across adjacent production layers; no model weights are needed here.

use std::time::Instant;

use glcuda::buffer::BackendBuffer;
use glcuda::driver::{cuda_available, Cuda};
use glcuda::kernels::KernelSet;

const ROWS: usize = 244;
const DIM: usize = 896;
const WARMUP: usize = 10;
const ITERS: usize = 100;
const REPEATS: usize = 5;

fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let unit =
                (state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32 / (1u64 << 24) as f32;
            unit - 0.5
        })
        .collect()
}

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
    Ok(start.elapsed().as_secs_f64() * 1.0e6 / ITERS as f64)
}

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(f64::total_cmp);
    (samples[samples.len() / 2 - 1] + samples[samples.len() / 2]) * 0.5
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        println!("[wave111-residual-fusion] no CUDA device; nothing measured");
        return Ok(());
    }
    let cuda = Cuda::probe()?;
    let kernels = KernelSet::load(&cuda)?;
    let elems = ROWS * DIM;
    let blocks = elems / 32;
    let x = values(elems, 111);
    let residual = values(elems, 112);
    let norm: Vec<f32> = values(DIM, 113)
        .into_iter()
        .map(|value| value * 0.25 + 1.0)
        .collect();

    let bytes =
        ((2 * elems + DIM + 2 * elems + 2 * elems + 2 * blocks) * 4 + 2 * elems + 1_048_576) as u64;
    let mut buffer = BackendBuffer::new(&cuda, bytes)?;
    let dx_retained = buffer.alloc_f32(elems)?.dptr;
    let dx_candidate = buffer.alloc_f32(elems)?.dptr;
    let dresidual = buffer.alloc_f32(elems)?.dptr;
    let dnorm = buffer.alloc_f32(DIM)?.dptr;
    let dout_retained = buffer.alloc_f32(elems)?.dptr;
    let dout_candidate = buffer.alloc_f32(elems)?.dptr;
    let dqs_retained = buffer.alloc(elems as u64)?.dptr;
    let dqs_candidate = buffer.alloc(elems as u64)?.dptr;
    let dscales_retained = buffer.alloc_f32(blocks)?.dptr;
    let dscales_candidate = buffer.alloc_f32(blocks)?.dptr;
    cuda.htod_f32(dx_retained, &x)?;
    cuda.htod_f32(dx_candidate, &x)?;
    cuda.htod_f32(dresidual, &residual)?;
    cuda.htod_f32(dnorm, &norm)?;

    let retained = || {
        kernels.add(&cuda, dx_retained, dresidual, elems as u32)?;
        kernels.rms_quantize_q8_rows(
            &cuda,
            dx_retained,
            None,
            dnorm,
            dout_retained,
            dqs_retained,
            dscales_retained,
            DIM as u32,
            1.0e-6,
            ROWS as u32,
        )
    };
    let candidate = || {
        kernels.rms_quantize_q8_rows(
            &cuda,
            dx_candidate,
            Some(dresidual),
            dnorm,
            dout_candidate,
            dqs_candidate,
            dscales_candidate,
            DIM as u32,
            1.0e-6,
            ROWS as u32,
        )
    };

    retained()?;
    candidate()?;
    cuda.synchronize()?;
    let mut xr = vec![0.0f32; elems];
    let mut xc = vec![0.0f32; elems];
    let mut outr = vec![0.0f32; elems];
    let mut outc = vec![0.0f32; elems];
    let mut qsr = vec![0.0f32; elems / 4];
    let mut qsc = vec![0.0f32; elems / 4];
    let mut scr = vec![0.0f32; blocks];
    let mut scc = vec![0.0f32; blocks];
    cuda.dtoh_f32(&mut xr, dx_retained)?;
    cuda.dtoh_f32(&mut xc, dx_candidate)?;
    cuda.dtoh_f32(&mut outr, dout_retained)?;
    cuda.dtoh_f32(&mut outc, dout_candidate)?;
    cuda.dtoh_f32(&mut qsr, dqs_retained)?;
    cuda.dtoh_f32(&mut qsc, dqs_candidate)?;
    cuda.dtoh_f32(&mut scr, dscales_retained)?;
    cuda.dtoh_f32(&mut scc, dscales_candidate)?;
    let x_exact = xr
        .iter()
        .map(|v| v.to_bits())
        .eq(xc.iter().map(|v| v.to_bits()));
    let out_exact = outr
        .iter()
        .map(|v| v.to_bits())
        .eq(outc.iter().map(|v| v.to_bits()));
    let q8_exact = qsr
        .iter()
        .map(|v| v.to_bits())
        .eq(qsc.iter().map(|v| v.to_bits()));
    let scales_exact = scr
        .iter()
        .map(|v| v.to_bits())
        .eq(scc.iter().map(|v| v.to_bits()));

    let (mut retained_samples, mut candidate_samples) = (
        Vec::with_capacity(2 * REPEATS),
        Vec::with_capacity(2 * REPEATS),
    );
    for _ in 0..REPEATS {
        retained_samples.push(timed(&cuda, retained)?);
        candidate_samples.push(timed(&cuda, candidate)?);
        candidate_samples.push(timed(&cuda, candidate)?);
        retained_samples.push(timed(&cuda, retained)?);
    }
    let retained_us = median(&mut retained_samples);
    let candidate_us = median(&mut candidate_samples);
    println!(
        "[wave111-direct] {{\"rows\":{},\"dim\":{},\"warmup\":{},\"iters\":{},\"repeats\":{},\"retained_us\":{:.3},\"candidate_us\":{:.3},\"speedup\":{:.4},\"x_bit_exact\":{},\"out_bit_exact\":{},\"q8_bit_exact\":{},\"scales_bit_exact\":{}}}",
        ROWS,
        DIM,
        WARMUP,
        ITERS,
        REPEATS,
        retained_us,
        candidate_us,
        retained_us / candidate_us,
        x_exact,
        out_exact,
        q8_exact,
        scales_exact,
    );
    buffer.free(&cuda)?;
    Ok(())
}
