//! Wave 64 compensated-MMA AV feasibility screen.
//!
//! This loads an isolated PTX module. It does not alter KernelSet or the
//! production attention dispatcher.

use std::ffi::c_void;
use std::time::Instant;

use glcuda::buffer::BackendBuffer;
use glcuda::driver::{cuda_available, Cuda, Kernel};
use glcuda::ffi::CUdeviceptr;

const HEADS: usize = 14;
const WIDTH: usize = 64;
const WARMUP: usize = 10;
const ITERS: usize = 100;
const REPEATS: usize = 5;
const MAX_ABS_GATE: f32 = 1.0e-5;
const MIN_PRODUCTION_SPEEDUP: f64 = 1.50;
const PTX: &str = include_str!("../src/kernels/glcuda_sm75_wave64.ptx");

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

fn causal_probabilities(ntok: usize, capacity: usize) -> Vec<f32> {
    let raw = values(HEADS * ntok * capacity, 64 + capacity as u64);
    let mut out = vec![0.0; raw.len()];
    for head in 0..HEADS {
        for row in 0..ntok {
            let visible = (row + 1).min(capacity);
            let base = (head * ntok + row) * capacity;
            let max = raw[base..base + visible]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for key in 0..visible {
                let weight = (raw[base + key] - max).exp();
                out[base + key] = weight;
                sum += weight;
            }
            for key in 0..visible {
                out[base + key] /= sum;
            }
        }
    }
    out
}

fn launch_av(
    cuda: &Cuda,
    kernel: Kernel,
    prob: CUdeviceptr,
    v: CUdeviceptr,
    out: CUdeviceptr,
    ntok: u32,
    capacity: u32,
) -> Result<(), glcore::GlError> {
    let (mut prob, mut v, mut out, mut ntok, mut capacity) = (prob, v, out, ntok, capacity);
    let mut params = [
        &mut prob as *mut _ as *mut c_void,
        &mut v as *mut _ as *mut c_void,
        &mut out as *mut _ as *mut c_void,
        &mut ntok as *mut _ as *mut c_void,
        &mut capacity as *mut _ as *mut c_void,
    ];
    cuda.launch(
        kernel,
        (ntok.div_ceil(16), HEADS as u32, 1),
        (128, 1, 1),
        0,
        &mut params,
    )
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

struct Record {
    capacity: usize,
    scalar_us: f64,
    mma_us: f64,
    max_abs: f32,
    rms: f64,
}

impl Record {
    fn json(&self) -> String {
        format!(
            "{{\"capacity\":{},\"scalar_us\":{:.3},\"mma_us\":{:.3},\
             \"speedup\":{:.4},\"max_abs\":{:.9e},\"rms\":{:.9e}}}",
            self.capacity,
            self.scalar_us,
            self.mma_us,
            self.scalar_us / self.mma_us,
            self.max_abs,
            self.rms,
        )
    }
}

fn screen(
    cuda: &Cuda,
    scalar: Kernel,
    mma: Kernel,
    capacity: usize,
) -> Result<Record, Box<dyn std::error::Error>> {
    let ntok = capacity;
    let prob = causal_probabilities(ntok, capacity);
    let v = values(HEADS * capacity * WIDTH, 6400 + capacity as u64);
    let output_len = HEADS * ntok * WIDTH;
    let bytes = ((prob.len() + v.len() + 2 * output_len) * 4 + 1_048_576) as u64;
    let mut buffer = BackendBuffer::new(cuda, bytes)?;
    let dprob = buffer.alloc_f32(prob.len())?.dptr;
    let dv = buffer.alloc_f32(v.len())?.dptr;
    let dscalar = buffer.alloc_f32(output_len)?.dptr;
    let dmma = buffer.alloc_f32(output_len)?.dptr;
    cuda.htod_f32(dprob, &prob)?;
    cuda.htod_f32(dv, &v)?;

    let scalar_launch = || {
        launch_av(
            cuda,
            scalar,
            dprob,
            dv,
            dscalar,
            ntok as u32,
            capacity as u32,
        )
    };
    let mma_launch = || launch_av(cuda, mma, dprob, dv, dmma, ntok as u32, capacity as u32);
    scalar_launch()?;
    mma_launch()?;
    cuda.synchronize()?;
    let mut scalar_host = vec![0.0; output_len];
    let mut mma_host = vec![0.0; output_len];
    cuda.dtoh_f32(&mut scalar_host, dscalar)?;
    cuda.dtoh_f32(&mut mma_host, dmma)?;
    if !scalar_host
        .iter()
        .chain(&mma_host)
        .all(|value| value.is_finite())
    {
        return Err(format!("non-finite Wave 64 output at capacity {capacity}").into());
    }
    let mut max_abs = 0.0f32;
    let mut squared = 0.0f64;
    for (&left, &right) in scalar_host.iter().zip(&mma_host) {
        let delta = (left - right).abs();
        max_abs = max_abs.max(delta);
        squared += f64::from(delta).powi(2);
    }
    let rms = (squared / output_len as f64).sqrt();

    let (mut scalar_samples, mut mma_samples) = (
        Vec::with_capacity(2 * REPEATS),
        Vec::with_capacity(2 * REPEATS),
    );
    for _ in 0..REPEATS {
        scalar_samples.push(timed(cuda, scalar_launch)?);
        mma_samples.push(timed(cuda, mma_launch)?);
        mma_samples.push(timed(cuda, mma_launch)?);
        scalar_samples.push(timed(cuda, scalar_launch)?);
    }
    let scalar_us = median(&mut scalar_samples);
    let mma_us = median(&mut mma_samples);
    buffer.free(cuda)?;
    Ok(Record {
        capacity,
        scalar_us,
        mma_us,
        max_abs,
        rms,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        println!("[wave64-av] no CUDA device; nothing measured");
        return Ok(());
    }
    let cuda = Cuda::probe()?;
    if (cuda.info.sm_major, cuda.info.sm_minor) != (7, 5) {
        return Err(format!(
            "Wave 64 requires sm_75, got sm_{}{}",
            cuda.info.sm_major, cuda.info.sm_minor
        )
        .into());
    }
    let module = cuda.load_module(PTX)?;
    let scalar = module.get_function("gl_wave64_av_scalar_f32")?;
    let mma = module.get_function("gl_wave75_av_mma4_row_f32")?;
    let records = [
        screen(&cuda, scalar, mma, 1)?,
        screen(&cuda, scalar, mma, 17)?,
        screen(&cuda, scalar, mma, 241)?,
        screen(&cuda, scalar, mma, 244)?,
    ];
    println!(
        "[wave64-resource] {{\"threads\":128,\"dynamic_shared_bytes\":0,\
         \"occupancy_source\":\"ptxas-only\"}}",
    );
    println!(
        "[wave64-av] {{\"warmup\":{},\"iters\":{},\"repeats\":{},\"records\":[{}]}}",
        WARMUP,
        ITERS,
        REPEATS,
        records
            .iter()
            .map(Record::json)
            .collect::<Vec<_>>()
            .join(","),
    );
    if let Some(record) = records.iter().find(|record| record.max_abs > MAX_ABS_GATE) {
        return Err(format!(
            "Wave 64 numeric gate failed at capacity {}: {:.9e} > {:.1e}",
            record.capacity, record.max_abs, MAX_ABS_GATE
        )
        .into());
    }
    let production = records.last().expect("records is non-empty");
    let speedup = production.scalar_us / production.mma_us;
    if speedup < MIN_PRODUCTION_SPEEDUP {
        return Err(format!(
            "Wave 64 speed gate failed: {speedup:.4}x < {MIN_PRODUCTION_SPEEDUP:.2}x"
        )
        .into());
    }
    Ok(())
}
