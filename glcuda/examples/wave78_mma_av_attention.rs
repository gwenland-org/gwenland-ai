//! Wave 78 direct gate for compensated-MMA AV inside fused production attention.
//!
//! This is a correctness/resource/speed screen against retained `mma4-regq`.
//! It is not production retention evidence; only `glbench` can provide that.

use std::mem::size_of;
use std::time::Instant;

use glcuda::buffer::BackendBuffer;
use glcuda::driver::{cuda_available, Cuda};
use glcuda::kernels::KernelSet;

const N_HEADS: usize = 14;
const N_KV: usize = 2;
const HEAD_DIM: usize = 64;
const NTOK: usize = 244;
const HEAD_STRIDE: usize = 256 * HEAD_DIM;
const WIDTH: usize = N_HEADS * HEAD_DIM;
const WARMUP: usize = 20;
const ITERS: usize = 200;
const ROUNDS: usize = 7;
const MAX_ABS: f32 = 1.0e-5;
const MIN_DIRECT_SPEEDUP: f64 = 1.10;

fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            ((state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32 / (1u64 << 24) as f32 - 0.5)
                * 2.0
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
    Ok(start.elapsed().as_secs_f64() * 1e6 / ITERS as f64)
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        return Err("Wave 78 direct gate requires a CUDA device".into());
    }
    let cuda = Cuda::probe()?;
    if (cuda.info.sm_major, cuda.info.sm_minor) != (7, 5) {
        return Err(format!(
            "Wave 78 direct gate requires sm_75, got sm_{}{}",
            cuda.info.sm_major, cuda.info.sm_minor
        )
        .into());
    }
    let kernels = KernelSet::load(&cuda)?;
    if !kernels.has_mma() {
        return Err("Wave 78 requires the sm_75 MMA module".into());
    }

    let entries = kernels.attention_entries(NTOK as u32);
    let (retained_kernel, retained_shared) = entries
        .iter()
        .find(|(name, _, _)| *name == "mma4_regq")
        .map(|(_, kernel, shared)| (*kernel, *shared))
        .ok_or("Wave 48 retained resource entry unavailable")?;
    let (candidate_kernel, candidate_shared) = entries
        .iter()
        .find(|(name, _, _)| *name == "mma4_regq_avmma")
        .map(|(_, kernel, shared)| (*kernel, *shared))
        .ok_or("Wave 78 candidate resource entry unavailable")?;
    let retained_blocks = cuda
        .max_active_blocks_per_sm(retained_kernel, 128, retained_shared as usize)
        .ok_or("Wave 48 occupancy query unavailable")?;
    let candidate_blocks = cuda
        .max_active_blocks_per_sm(candidate_kernel, 128, candidate_shared as usize)
        .ok_or("Wave 78 occupancy query unavailable")?;

    let q = values(NTOK * WIDTH, 78);
    let k = values(N_KV * HEAD_STRIDE, 79);
    let v = values(N_KV * HEAD_STRIDE, 80);
    let pos: Vec<u32> = (0..=NTOK as u32).collect();
    let bytes =
        ((q.len() + k.len() + v.len() + 2 * NTOK * WIDTH) * 4 + pos.len() * 4 + 1_048_576) as u64;
    let mut buffer = BackendBuffer::new(&cuda, bytes)?;
    let dq = buffer.alloc_f32(q.len())?.dptr;
    let dk = buffer.alloc_f32(k.len())?.dptr;
    let dv = buffer.alloc_f32(v.len())?.dptr;
    let dpos = buffer.alloc((pos.len() * 4) as u64)?.dptr;
    let retained_out = buffer.alloc_f32(NTOK * WIDTH)?.dptr;
    let candidate_out = buffer.alloc_f32(NTOK * WIDTH)?.dptr;
    cuda.htod_f32(dq, &q)?;
    cuda.htod_f32(dk, &k)?;
    cuda.htod_f32(dv, &v)?;
    let pos_bytes = unsafe {
        std::slice::from_raw_parts(pos.as_ptr().cast::<u8>(), pos.len() * size_of::<u32>())
    };
    cuda.htod(dpos, pos_bytes)?;

    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let retained = || {
        kernels.attn_mma4_regq_fused(
            &cuda,
            dq,
            dk,
            dv,
            retained_out,
            N_HEADS as u32,
            HEAD_DIM as u32,
            dpos,
            (N_HEADS / N_KV) as u32,
            HEAD_STRIDE as u32,
            scale,
            NTOK as u32,
            NTOK as u32,
            WIDTH as u32,
        )
    };
    let candidate = || {
        kernels.attn_mma4_regq_avmma_fused(
            &cuda,
            dq,
            dk,
            dv,
            candidate_out,
            N_HEADS as u32,
            HEAD_DIM as u32,
            dpos,
            (N_HEADS / N_KV) as u32,
            HEAD_STRIDE as u32,
            scale,
            NTOK as u32,
            NTOK as u32,
            WIDTH as u32,
        )
    };

    retained()?;
    candidate()?;
    cuda.synchronize()?;
    let mut retained_host = vec![0.0f32; NTOK * WIDTH];
    let mut candidate_host = vec![0.0f32; NTOK * WIDTH];
    cuda.dtoh_f32(&mut retained_host, retained_out)?;
    cuda.dtoh_f32(&mut candidate_host, candidate_out)?;
    let mut max_abs = 0.0f32;
    let mut sq = 0.0f64;
    for (&left, &right) in retained_host.iter().zip(&candidate_host) {
        if !right.is_finite() {
            return Err("Wave 78 candidate produced a non-finite value".into());
        }
        let diff = (left - right).abs();
        max_abs = max_abs.max(diff);
        sq += f64::from(diff) * f64::from(diff);
    }
    let rms = (sq / retained_host.len() as f64).sqrt();
    if max_abs > MAX_ABS {
        return Err(
            format!("Wave 78 numeric gate failed: max_abs={max_abs:.9e} > {MAX_ABS:.1e}").into(),
        );
    }

    let mut retained_us = Vec::with_capacity(ROUNDS);
    let mut candidate_us = Vec::with_capacity(ROUNDS);
    for round in 0..ROUNDS {
        if round % 2 == 0 {
            retained_us.push(timed(&cuda, retained)?);
            candidate_us.push(timed(&cuda, candidate)?);
        } else {
            candidate_us.push(timed(&cuda, candidate)?);
            retained_us.push(timed(&cuda, retained)?);
        }
    }
    let retained_us = median(retained_us);
    let candidate_us = median(candidate_us);
    let speedup = retained_us / candidate_us;
    println!(
        "[wave78-direct] {{\"ntok\":{NTOK},\"heads\":{N_HEADS},\"kv_heads\":{N_KV},\"head_dim\":{HEAD_DIM},\"warmup\":{WARMUP},\"iters\":{ITERS},\"rounds\":{ROUNDS},\"retained_us\":{retained_us:.3},\"candidate_us\":{candidate_us:.3},\"speedup\":{speedup:.4},\"max_abs\":{max_abs:.9e},\"rms\":{rms:.9e},\"retained_dynamic_shared\":{retained_shared},\"candidate_dynamic_shared\":{candidate_shared},\"retained_blocks_per_sm\":{retained_blocks},\"candidate_blocks_per_sm\":{candidate_blocks}}}"
    );
    buffer.free(&cuda)?;
    if speedup < MIN_DIRECT_SPEEDUP {
        return Err(format!(
            "Wave 78 direct speed gate failed: {speedup:.4}x < {MIN_DIRECT_SPEEDUP:.2}x"
        )
        .into());
    }
    Ok(())
}
