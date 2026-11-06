//! A measured read-bandwidth figure for the device.
//!
//! glbench judges decode against a bandwidth ceiling, and a vendor spec sheet
//! is a promise rather than a measurement: real parts sustain less than their
//! advertised peak. This streams a buffer far larger than the L2 through a read
//! kernel and reports the best rate seen, the GPU counterpart of the host read
//! probe in `glbench::environment::bandwidth`.
//!
//! Read-only on purpose. A device-to-device copy would be simpler, but decode
//! streams weights and writes almost nothing, and a copy's write traffic and
//! bus turnarounds would measure a different ceiling.

use std::ffi::c_void;

use glcore::GlError;

use crate::driver::Cuda;

const PTX: &str = include_str!("kernels/glcuda_bandwidth.ptx");
const ENTRY: &str = "gl_read_bandwidth";

/// Threads per block.
const BLOCK: u32 = 256;
/// Blocks per SM: enough resident warps to keep the memory system saturated.
const BLOCKS_PER_SM: u32 = 8;
/// Bytes one thread reads per round: four 16-byte vectors.
const BYTES_PER_THREAD_ROUND: u64 = 64;
/// Target buffer size, far beyond any L2 so the rate cannot be a cache's.
const TARGET_BYTES: u64 = 512 << 20;
/// Untimed passes, to bring the clocks up before anything is recorded.
const WARMUP_PASSES: usize = 3;
/// Timed passes; the best is reported, since a ceiling is what the device can
/// reach, and interference only ever lowers a pass.
const TIMED_PASSES: usize = 8;

/// Launch geometry and the exact buffer size it reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Plan {
    grid: u32,
    rounds: u32,
    bytes: u64,
}

/// Size the launch for `sm_count` SMs. The buffer is a whole number of rounds,
/// which is what lets the kernel run without a bounds check.
fn plan(sm_count: u32) -> Plan {
    let grid = sm_count.max(1) * BLOCKS_PER_SM;
    let per_round = u64::from(grid) * u64::from(BLOCK) * BYTES_PER_THREAD_ROUND;
    let rounds = (TARGET_BYTES / per_round).max(1);
    Plan { grid, rounds: rounds as u32, bytes: per_round * rounds }
}

/// Measure device read bandwidth in GB/s (10^9 bytes per second).
///
/// Costs well under a second. Run it before an engine loads: it needs a free
/// 512 MiB and must not share the device with a measured window.
pub fn measure_read_gbs(cuda: &Cuda) -> Result<f64, GlError> {
    let plan = plan(cuda.info.sm_count.max(1) as u32);
    let (free, _) = cuda.mem_get_info()?;
    if (free as u64) < plan.bytes + (64 << 20) {
        return Err(GlError::Engine(format!(
            "bandwidth probe needs {} MiB of free VRAM, {} MiB available",
            plan.bytes >> 20,
            free >> 20
        )));
    }
    let ring = cuda
        .event_ring(2)
        .ok_or_else(|| GlError::Engine("CUDA events unavailable for the bandwidth probe".into()))?;
    let module = cuda.load_module(PTX)?;
    let kernel = module.get_function(ENTRY)?;

    let src = cuda.mem_alloc(plan.bytes as usize)?;
    let out = match cuda.mem_alloc(4) {
        Ok(out) => out,
        Err(error) => {
            let _ = cuda.mem_free(src);
            return Err(error);
        }
    };

    let result = (|| {
        let (mut p_src, mut p_out, mut p_rounds) = (src, out, plan.rounds);
        let mut params: [*mut c_void; 3] = [
            (&mut p_src as *mut crate::ffi::CUdeviceptr).cast(),
            (&mut p_out as *mut crate::ffi::CUdeviceptr).cast(),
            (&mut p_rounds as *mut u32).cast(),
        ];
        let mut best_ms = f64::INFINITY;
        for pass in 0..WARMUP_PASSES + TIMED_PASSES {
            ring.record(cuda, 0);
            cuda.launch(kernel, (plan.grid, 1, 1), (BLOCK, 1, 1), 0, &mut params)?;
            ring.record(cuda, 1);
            let ms = ring.elapsed_ms(0, 1).ok_or_else(|| {
                GlError::Engine("bandwidth probe could not read its timing events".into())
            })?;
            if pass >= WARMUP_PASSES && ms > 0.0 {
                best_ms = best_ms.min(ms);
            }
        }
        if best_ms.is_finite() {
            Ok(plan.bytes as f64 / (best_ms * 1e6))
        } else {
            Err(GlError::Engine("bandwidth probe recorded no timed pass".into()))
        }
    })();

    let _ = cuda.mem_free(out);
    let _ = cuda.mem_free(src);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_plan_reads_a_whole_number_of_rounds_past_any_l2() {
        for sm_count in [1, 40, 46, 108, 128] {
            let p = plan(sm_count);
            let per_round = u64::from(p.grid) * u64::from(BLOCK) * BYTES_PER_THREAD_ROUND;
            assert_eq!(p.bytes, per_round * u64::from(p.rounds));
            assert!(p.bytes >= 256 << 20, "{sm_count} SMs: {} bytes", p.bytes);
            assert!(p.bytes <= TARGET_BYTES);
        }
    }

    /// ptxas rejects the whole module over one stray byte, and it would only
    /// surface on a machine with a GPU.
    #[test]
    fn the_ptx_is_pure_ascii_with_lf_endings() {
        assert!(PTX.is_ascii());
        assert!(!PTX.contains('\r'));
        assert!(PTX.contains(ENTRY));
    }
}
