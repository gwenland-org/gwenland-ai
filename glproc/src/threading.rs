//! Persistent worker pool with contiguous per-thread row chunks.
//!
//! Why a persistent pool? `thread::spawn` costs tens of microseconds; a
//! decode step dispatches ~170 matvecs, so spawning per call would burn the
//! whole latency budget. Workers are spawned once and parked on a condvar.
//!
//! Why contiguous chunks (thread `t` takes rows `[t*chunk, (t+1)*chunk)`)
//! instead of interleaved rows? Row cost is uniform so both balance
//! perfectly, but the weights stream from DRAM every token (they cannot fit
//! in cache), and single-channel DDR4 rewards a few clean sequential
//! streams with far better page locality than N interleaved streams that
//! each skip (N−1) rows between reads. Measured on the i3-1115G4: chunked
//! beat interleaved by ~35% end-to-end (18.6 → 25.3 tok/s on
//! Qwen2.5-0.5B), with the Q8_0 lm-head matvec reaching ~23 GB/s.
//!
//! Layer-level parallelism (thread 0 runs layers 0,4,8...) is impossible for
//! autoregressive decode — layer L+1 consumes layer L's output — so the
//! parallel axis here is the *rows within each matvec*, which have no data
//! dependencies at all.
//!
//! Fallback: if worker spawn fails, the pool degrades to 0 workers and every
//! `run` executes single-threaded on the caller. No panic.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

use crate::kernels::bridge::{bridge_row_dot, QuantFormat};
use crate::kernels::matmul;
use crate::kernels::qdot::{row_dot_q8, row_dot_q8_xn, QuantizedActivation};
use crate::simd_strategy::SimdStrategy;

/// Type-erased pointer to the job closure. Only valid while `ThreadPool::run`
/// is blocked — `run` does not return until every worker has finished.
#[derive(Clone, Copy)]
struct JobPtr(*const (dyn Fn(usize) + Sync));

// SAFETY: the pointee is `Sync` (shared-callable from any thread) and its
// lifetime is enforced by `run` blocking until all workers are done with it.
unsafe impl Send for JobPtr {}

/// Iterations a worker spins on the generation counter before parking on
/// the condvar. A decode step dispatches ~170 jobs back-to-back with only
/// microseconds between them; spinning bridges those gaps so the kernel
/// scheduler (each wake ≈ 5–50 µs) stays out of the hot path. ~2^14 pause
/// iterations ≈ 10–20 µs — long enough to catch the next matvec, short
/// enough not to burn a core when generation stops.
const SPIN_ITERS: u32 = 1 << 14;

struct PoolShared {
    /// Bumped once per dispatched job (after `job` is written). Workers spin
    /// on this — one atomic load per iteration, no lock.
    generation: AtomicU64,
    /// Workers still running the current job.
    remaining: AtomicUsize,
    shutdown: AtomicBool,
    /// The current job. Written by `run` before the generation bump
    /// (release) and read by workers after observing the bump (acquire),
    /// so the plain cell access is ordered — see SAFETY notes at the uses.
    job: UnsafeCell<Option<JobPtr>>,
    /// Parking lot for workers whose spin budget ran out.
    lock: Mutex<()>,
    work_cv: Condvar,
    /// Signals the caller: all workers finished the current job.
    done_cv: Condvar,
}

// SAFETY: `job` is the only non-Sync field; its cross-thread handoff is
// ordered by the acquire/release pair on `generation` (write-before-bump,
// read-after-observe) and `run` never overwrites it while workers run.
unsafe impl Sync for PoolShared {}

/// Fixed-size worker pool. The calling thread participates as thread 0, so
/// `ThreadPool::new(4)` spawns 3 workers.
pub struct ThreadPool {
    shared: Arc<PoolShared>,
    workers: Vec<JoinHandle<()>>,
    n_threads: usize,
}

impl ThreadPool {
    /// Create a pool of `n_threads` total executors (including the caller).
    /// If any worker fails to spawn, the pool silently degrades toward
    /// single-threaded execution instead of panicking.
    pub fn new(n_threads: usize) -> Self {
        let n_threads = n_threads.max(1);
        let shared = Arc::new(PoolShared {
            generation: AtomicU64::new(0),
            remaining: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
            job: UnsafeCell::new(None),
            lock: Mutex::new(()),
            work_cv: Condvar::new(),
            done_cv: Condvar::new(),
        });

        let mut workers = Vec::with_capacity(n_threads - 1);
        for tid in 1..n_threads {
            let sh = Arc::clone(&shared);
            let spawned = std::thread::Builder::new()
                .name(format!("glproc-worker-{tid}"))
                .spawn(move || worker_loop(sh, tid));
            match spawned {
                Ok(handle) => workers.push(handle),
                // Spawn failure = fewer workers; run() still works.
                Err(_) => break,
            }
        }

        let n_threads = workers.len() + 1;
        ThreadPool {
            shared,
            workers,
            n_threads,
        }
    }

    /// Total executor count (workers + calling thread).
    pub fn n_threads(&self) -> usize {
        self.n_threads
    }

    /// Execute `f(thread_id)` on every thread (ids `0..n_threads`) and wait
    /// for all of them. The caller runs `f(0)` itself.
    pub fn run(&self, f: &(dyn Fn(usize) + Sync)) {
        if self.workers.is_empty() {
            f(0);
            return;
        }

        // SAFETY: we erase the borrow's lifetime to hand it to the workers,
        // but this function blocks until `remaining == 0`, i.e. until every
        // worker has returned from `f` — so the pointee outlives every use.
        let job = JobPtr(unsafe {
            std::mem::transmute::<&(dyn Fn(usize) + Sync), &'static (dyn Fn(usize) + Sync)>(f)
                as *const _
        });

        // SAFETY: no worker reads `job` until it observes the generation
        // bump below, and the previous job's readers are all gone
        // (`remaining` reached 0 before the last `run` returned).
        unsafe { *self.shared.job.get() = Some(job) };
        // Arm `remaining` before the bump: a spinning worker acts on the
        // bump immediately and must find the counter already set.
        self.shared
            .remaining
            .store(self.workers.len(), Ordering::Relaxed);
        self.shared.generation.fetch_add(1, Ordering::Release);
        // Wake any parked workers. Taking the lock orders this notify
        // against a worker sitting between its last generation check and
        // the condvar wait — without it the wakeup could be lost.
        {
            let _g = self.shared.lock.lock().unwrap();
            self.shared.work_cv.notify_all();
        }

        // The caller is thread 0 — do its share instead of just waiting.
        f(0);

        // Workers usually finish within the caller's own share; spin for
        // the stragglers and only park when the budget runs out.
        let mut spins = 0u32;
        while self.shared.remaining.load(Ordering::Acquire) > 0 {
            if spins < SPIN_ITERS {
                std::hint::spin_loop();
                spins += 1;
            } else {
                let mut g = self.shared.lock.lock().unwrap();
                while self.shared.remaining.load(Ordering::Acquire) > 0 {
                    g = self.shared.done_cv.wait(g).unwrap();
                }
                break;
            }
        }
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::Release);
        {
            let _g = self.shared.lock.lock().unwrap();
            self.shared.work_cv.notify_all();
        }
        for handle in self.workers.drain(..) {
            let _ = handle.join();
        }
    }
}

fn worker_loop(shared: Arc<PoolShared>, tid: usize) {
    let mut seen_generation = 0u64;
    loop {
        // Spin for the next generation bump; park once the budget runs out.
        let mut spins = 0u32;
        loop {
            if shared.shutdown.load(Ordering::Acquire) {
                return;
            }
            let generation = shared.generation.load(Ordering::Acquire);
            if generation != seen_generation {
                seen_generation = generation;
                break;
            }
            if spins < SPIN_ITERS {
                std::hint::spin_loop();
                spins += 1;
            } else {
                let mut g = shared.lock.lock().unwrap();
                loop {
                    if shared.shutdown.load(Ordering::Acquire) {
                        return;
                    }
                    let generation = shared.generation.load(Ordering::Acquire);
                    if generation != seen_generation {
                        seen_generation = generation;
                        break;
                    }
                    g = shared.work_cv.wait(g).unwrap();
                }
                break;
            }
        }

        // SAFETY: the Acquire load of `generation` above synchronizes with
        // `run`'s Release bump, which happens after `job` was written — so
        // the cell holds the current job and nobody writes it while we read.
        let job = unsafe { (*shared.job.get()).expect("job set before generation bump") };

        // SAFETY: `ThreadPool::run` keeps the closure alive until we
        // decrement `remaining` below, and the closure is `Sync`.
        unsafe { (*job.0)(tid) };

        // Release so the caller's Acquire load of `remaining == 0` also
        // publishes this worker's output-row writes.
        if shared.remaining.fetch_sub(1, Ordering::AcqRel) == 1 {
            let _g = shared.lock.lock().unwrap();
            shared.done_cv.notify_all();
        }
    }
}

/// Output slice handed to multiple threads. Each thread writes a disjoint
/// row range, so there is never a data race.
struct RowWriter(*mut f32);

// SAFETY: threads write disjoint index ranges (contiguous chunks) and
// nobody reads until the pool's barrier in `run` has passed.
unsafe impl Send for RowWriter {}
unsafe impl Sync for RowWriter {}

/// Threaded f32 matvec: `y[o] = dot(w[o], x)`, rows interleaved across the
/// pool. `w` is `[out_dim, in_dim]` row-major, `y.len() == out_dim`.
pub fn par_matvec(
    pool: &ThreadPool,
    w: &[f32],
    x: &[f32],
    y: &mut [f32],
    out_dim: usize,
    in_dim: usize,
    strategy: SimdStrategy,
) {
    debug_assert_eq!(w.len(), out_dim * in_dim);
    debug_assert_eq!(y.len(), out_dim);
    let n = pool.n_threads();
    let out = RowWriter(y.as_mut_ptr());
    // Contiguous chunk per thread — sequential weight streams beat
    // interleaved rows on single-channel DRAM (see par_matvec_qdot).
    let chunk = out_dim.div_ceil(n);

    pool.run(&|tid| {
        let out = &out;
        let lo = (tid * chunk).min(out_dim);
        let hi = (lo + chunk).min(out_dim);
        for o in lo..hi {
            let row = &w[o * in_dim..(o + 1) * in_dim];
            // SAFETY: `strategy` comes from SimdStrategy::detect(); rows are
            // disjoint per thread (contiguous chunks).
            let dot = match strategy {
                SimdStrategy::Avx512 => unsafe { matmul::avx512::dot_f32(row, x) },
                SimdStrategy::Avx2 => unsafe { matmul::avx2::dot_f32(row, x) },
                SimdStrategy::Scalar => matmul::scalar::dot_f32(row, x),
            };
            // SAFETY: o < out_dim == y.len(), and no other thread touches o.
            unsafe { *out.0.add(o) = dot };
        }
    });
}

/// Threaded quantized bridge matvec: each row is dequantized block-by-block
/// into a stack buffer and dotted while L1-hot. Rows interleaved across the
/// pool. Works for any [`QuantFormat`].
pub fn par_matvec_quant(
    pool: &ThreadPool,
    fmt: QuantFormat,
    weights: &[u8],
    x: &[f32],
    y: &mut [f32],
    out_dim: usize,
    in_dim: usize,
    strategy: SimdStrategy,
) {
    debug_assert_eq!(in_dim % fmt.block_numel(), 0);
    debug_assert_eq!(y.len(), out_dim);
    let row_bytes = in_dim / fmt.block_numel() * fmt.block_bytes();
    debug_assert_eq!(weights.len(), out_dim * row_bytes);
    let n = pool.n_threads();
    let out = RowWriter(y.as_mut_ptr());
    // Contiguous chunk per thread — see par_matvec_qdot.
    let chunk = out_dim.div_ceil(n);

    pool.run(&|tid| {
        let out = &out;
        let lo = (tid * chunk).min(out_dim);
        let hi = (lo + chunk).min(out_dim);
        for o in lo..hi {
            let row = &weights[o * row_bytes..(o + 1) * row_bytes];
            let dot = bridge_row_dot(fmt, row, x, strategy);
            // SAFETY: o < out_dim == y.len(), and no other thread touches o.
            unsafe { *out.0.add(o) = dot };
        }
    });
}

/// Threaded integer-domain matvec: quantized weights × Q8 activation, rows
/// interleaved across the pool. The activation must already be quantized
/// (once per matvec, by the caller) for `in_dim` elements.
pub fn par_matvec_qdot(
    pool: &ThreadPool,
    fmt: QuantFormat,
    weights: &[u8],
    act: &QuantizedActivation,
    y: &mut [f32],
    out_dim: usize,
    in_dim: usize,
    strategy: SimdStrategy,
) {
    debug_assert_eq!(in_dim % fmt.block_numel(), 0);
    debug_assert_eq!(act.len, in_dim);
    debug_assert_eq!(y.len(), out_dim);
    let row_bytes = in_dim / fmt.block_numel() * fmt.block_bytes();
    debug_assert_eq!(weights.len(), out_dim * row_bytes);
    let n = pool.n_threads();
    let out = RowWriter(y.as_mut_ptr());
    // Contiguous chunk per thread, not interleaved rows: each thread then
    // reads one clean sequential weight stream, which single-channel DDR4
    // rewards with far better DRAM page locality than 4 interleaved streams
    // that each skip (n-1) rows between reads.
    let chunk = out_dim.div_ceil(n);

    pool.run(&|tid| {
        let out = &out;
        let lo = (tid * chunk).min(out_dim);
        let hi = (lo + chunk).min(out_dim);
        for o in lo..hi {
            let row = &weights[o * row_bytes..(o + 1) * row_bytes];
            let dot = row_dot_q8(fmt, row, act, strategy);
            // SAFETY: o < out_dim == y.len(), and no other thread touches o.
            unsafe { *out.0.add(o) = dot };
        }
    });
}

/// Fused SwiGLU matvec over row-interleaved weights:
/// `y[o] = silu(gate[o]·act) * (up[o]·act)`, where `packed` holds
/// `[gate row 0][up row 0][gate row 1]…` (see [`crate::model::GateUp`]).
///
/// One pool dispatch instead of three passes (gate matvec, up matvec,
/// activation sweep): both dots for a row live in registers until the
/// activated product is stored once — the intermediate gate/up vectors
/// never round-trip through RAM. The interleaved layout keeps each
/// thread's weight reads a single contiguous DRAM stream, and the two dot
/// chains per row are independent, doubling instruction-level parallelism.
pub fn par_matvec_swiglu(
    pool: &ThreadPool,
    fmt: QuantFormat,
    packed: &[u8],
    act: &QuantizedActivation,
    y: &mut [f32],
    out_dim: usize,
    in_dim: usize,
    strategy: SimdStrategy,
) {
    debug_assert_eq!(in_dim % fmt.block_numel(), 0);
    debug_assert_eq!(act.len, in_dim);
    debug_assert_eq!(y.len(), out_dim);
    let row_bytes = in_dim / fmt.block_numel() * fmt.block_bytes();
    debug_assert_eq!(packed.len(), out_dim * 2 * row_bytes);
    let n = pool.n_threads();
    let out = RowWriter(y.as_mut_ptr());
    // Contiguous chunk per thread — see par_matvec_qdot.
    let chunk = out_dim.div_ceil(n);

    pool.run(&|tid| {
        let out = &out;
        let lo = (tid * chunk).min(out_dim);
        let hi = (lo + chunk).min(out_dim);
        for o in lo..hi {
            let pair = &packed[o * 2 * row_bytes..(o + 1) * 2 * row_bytes];
            let g = row_dot_q8(fmt, &pair[..row_bytes], act, strategy);
            let u = row_dot_q8(fmt, &pair[row_bytes..], act, strategy);
            let s = g / (1.0 + crate::kernels::fast_exp(-g)) * u;
            // SAFETY: o < out_dim == y.len(), and no other thread touches o.
            unsafe { *out.0.add(o) = s };
        }
    });
}

/// Threaded f32 matmul for batched prefill: `y[b][col_off + o] = dot(w[o],
/// xb[b])`. `xb` holds `batch` input rows spaced `x_stride` apart (first
/// `in_dim` of each valid); `y` holds `batch` output rows spaced `y_stride`
/// apart. Threads chunk the *weight rows*, and each row is dotted with every
/// batch input while it is cache-hot — the weight matrix streams from DRAM
/// once per chunk instead of once per token, which is the entire point of
/// batching prefill.
#[allow(clippy::too_many_arguments)]
pub fn par_matmul(
    pool: &ThreadPool,
    w: &[f32],
    xb: &[f32],
    x_stride: usize,
    y: &mut [f32],
    y_stride: usize,
    col_off: usize,
    out_dim: usize,
    in_dim: usize,
    batch: usize,
    strategy: SimdStrategy,
) {
    debug_assert_eq!(w.len(), out_dim * in_dim);
    debug_assert!(xb.len() >= (batch - 1) * x_stride + in_dim);
    debug_assert!(y.len() >= (batch - 1) * y_stride + col_off + out_dim);
    let n = pool.n_threads();
    let out = RowWriter(y.as_mut_ptr());
    let chunk = out_dim.div_ceil(n);

    pool.run(&|tid| {
        let out = &out;
        let lo = (tid * chunk).min(out_dim);
        let hi = (lo + chunk).min(out_dim);
        for o in lo..hi {
            let row = &w[o * in_dim..(o + 1) * in_dim];
            for b in 0..batch {
                let x = &xb[b * x_stride..b * x_stride + in_dim];
                // SAFETY: strategy comes from SimdStrategy::detect().
                let dot = match strategy {
                    SimdStrategy::Avx512 => unsafe { matmul::avx512::dot_f32(row, x) },
                    SimdStrategy::Avx2 => unsafe { matmul::avx2::dot_f32(row, x) },
                    SimdStrategy::Scalar => matmul::scalar::dot_f32(row, x),
                };
                // SAFETY: (b, col_off + o) cells are disjoint across threads
                // (each thread owns a row range o) and within bounds per the
                // debug_assert above.
                unsafe { *out.0.add(b * y_stride + col_off + o) = dot };
            }
        }
    });
}

/// Threaded integer-domain matmul: quantized weights × a batch of Q8
/// activations. Same row-chunk/batch-inner structure as [`par_matmul`];
/// `acts[b]` must each be quantized for `in_dim` elements.
#[allow(clippy::too_many_arguments)]
pub fn par_matmul_qdot(
    pool: &ThreadPool,
    fmt: QuantFormat,
    weights: &[u8],
    acts: &[QuantizedActivation],
    y: &mut [f32],
    y_stride: usize,
    col_off: usize,
    out_dim: usize,
    in_dim: usize,
    strategy: SimdStrategy,
) {
    let batch = acts.len();
    debug_assert!(batch > 0);
    debug_assert_eq!(in_dim % fmt.block_numel(), 0);
    debug_assert!(acts.iter().all(|a| a.len == in_dim));
    let row_bytes = in_dim / fmt.block_numel() * fmt.block_bytes();
    debug_assert_eq!(weights.len(), out_dim * row_bytes);
    debug_assert!(y.len() >= (batch - 1) * y_stride + col_off + out_dim);
    let n = pool.n_threads();
    let out = RowWriter(y.as_mut_ptr());
    let chunk = out_dim.div_ceil(n);

    pool.run(&|tid| {
        let out = &out;
        let lo = (tid * chunk).min(out_dim);
        let hi = (lo + chunk).min(out_dim);
        for o in lo..hi {
            let row = &weights[o * row_bytes..(o + 1) * row_bytes];
            // Groups share the row's weight-side work (loads, sign prep,
            // f16 scale conversion) in one kernel call — 8 wide, then 4,
            // then singles for the remainder.
            let mut b = 0;
            while b + 8 <= batch {
                let group: [&QuantizedActivation; 8] = std::array::from_fn(|g| &acts[b + g]);
                let dots = row_dot_q8_xn::<8>(fmt, row, group, strategy);
                for (g, dot) in dots.into_iter().enumerate() {
                    // SAFETY: disjoint (b, col_off + o) cells per thread, in
                    // bounds per the debug_assert above.
                    unsafe { *out.0.add((b + g) * y_stride + col_off + o) = dot };
                }
                b += 8;
            }
            if b + 4 <= batch {
                let group: [&QuantizedActivation; 4] = std::array::from_fn(|g| &acts[b + g]);
                let dots = row_dot_q8_xn::<4>(fmt, row, group, strategy);
                for (g, dot) in dots.into_iter().enumerate() {
                    // SAFETY: as above.
                    unsafe { *out.0.add((b + g) * y_stride + col_off + o) = dot };
                }
                b += 4;
            }
            while b < batch {
                let dot = row_dot_q8(fmt, row, &acts[b], strategy);
                // SAFETY: as above.
                unsafe { *out.0.add(b * y_stride + col_off + o) = dot };
                b += 1;
            }
        }
    });
}

/// Batched fused SwiGLU matmul over row-interleaved gate/up weights:
/// `y[b][o] = silu(gate[o]·acts[b]) * (up[o]·acts[b])`. The batched analog
/// of [`par_matvec_swiglu`] — both weight rows of a pair are dotted with
/// every batch activation while cache-hot.
#[allow(clippy::too_many_arguments)]
pub fn par_matmul_swiglu(
    pool: &ThreadPool,
    fmt: QuantFormat,
    packed: &[u8],
    acts: &[QuantizedActivation],
    y: &mut [f32],
    y_stride: usize,
    out_dim: usize,
    in_dim: usize,
    strategy: SimdStrategy,
) {
    let batch = acts.len();
    debug_assert!(batch > 0);
    debug_assert_eq!(in_dim % fmt.block_numel(), 0);
    debug_assert!(acts.iter().all(|a| a.len == in_dim));
    let row_bytes = in_dim / fmt.block_numel() * fmt.block_bytes();
    debug_assert_eq!(packed.len(), out_dim * 2 * row_bytes);
    debug_assert!(y.len() >= (batch - 1) * y_stride + out_dim);
    let n = pool.n_threads();
    let out = RowWriter(y.as_mut_ptr());
    let chunk = out_dim.div_ceil(n);

    pool.run(&|tid| {
        let out = &out;
        let lo = (tid * chunk).min(out_dim);
        let hi = (lo + chunk).min(out_dim);
        for o in lo..hi {
            let pair = &packed[o * 2 * row_bytes..(o + 1) * 2 * row_bytes];
            let mut b = 0;
            while b + 8 <= batch {
                let group: [&QuantizedActivation; 8] = std::array::from_fn(|g| &acts[b + g]);
                let gs = row_dot_q8_xn::<8>(fmt, &pair[..row_bytes], group, strategy);
                let us = row_dot_q8_xn::<8>(fmt, &pair[row_bytes..], group, strategy);
                for g in 0..8 {
                    let s = gs[g] / (1.0 + crate::kernels::fast_exp(-gs[g])) * us[g];
                    // SAFETY: disjoint (b, o) cells per thread, in bounds
                    // per the debug_assert above.
                    unsafe { *out.0.add((b + g) * y_stride + o) = s };
                }
                b += 8;
            }
            if b + 4 <= batch {
                let group: [&QuantizedActivation; 4] = std::array::from_fn(|g| &acts[b + g]);
                let gs = row_dot_q8_xn::<4>(fmt, &pair[..row_bytes], group, strategy);
                let us = row_dot_q8_xn::<4>(fmt, &pair[row_bytes..], group, strategy);
                for g in 0..4 {
                    let s = gs[g] / (1.0 + crate::kernels::fast_exp(-gs[g])) * us[g];
                    // SAFETY: as above.
                    unsafe { *out.0.add((b + g) * y_stride + o) = s };
                }
                b += 4;
            }
            while b < batch {
                let g = row_dot_q8(fmt, &pair[..row_bytes], &acts[b], strategy);
                let u = row_dot_q8(fmt, &pair[row_bytes..], &acts[b], strategy);
                let s = g / (1.0 + crate::kernels::fast_exp(-g)) * u;
                // SAFETY: as above.
                unsafe { *out.0.add(b * y_stride + o) = s };
                b += 1;
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_runs_all_threads() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let pool = ThreadPool::new(4);
        let hits = AtomicUsize::new(0);
        pool.run(&|_tid| {
            hits.fetch_add(1, Ordering::SeqCst);
        });
        assert_eq!(hits.load(Ordering::SeqCst), pool.n_threads());
    }

    #[test]
    fn pool_reusable_across_jobs() {
        let pool = ThreadPool::new(3);
        for _ in 0..100 {
            use std::sync::atomic::{AtomicUsize, Ordering};
            let hits = AtomicUsize::new(0);
            pool.run(&|_| {
                hits.fetch_add(1, Ordering::SeqCst);
            });
            assert_eq!(hits.load(Ordering::SeqCst), pool.n_threads());
        }
    }

    #[test]
    fn single_thread_pool_runs_inline() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let pool = ThreadPool::new(1);
        let flag = AtomicBool::new(false);
        pool.run(&|tid| {
            assert_eq!(tid, 0);
            flag.store(true, Ordering::SeqCst);
        });
        assert!(flag.load(Ordering::SeqCst));
    }

    /// Synthetic Q8_0 rows: each 34-byte block is a 1.0 f16 scale (0x3C00)
    /// followed by 32 deterministic int8 quants.
    fn q8_0_weights(out_dim: usize, in_dim: usize) -> Vec<u8> {
        let blocks_per_row = in_dim / 32;
        let mut w = Vec::with_capacity(out_dim * blocks_per_row * 34);
        for o in 0..out_dim {
            for j in 0..blocks_per_row {
                w.extend_from_slice(&0x3C00u16.to_le_bytes()); // d = 1.0
                for i in 0..32 {
                    w.push(((o * 7 + j * 3 + i * 11) % 251) as i8 as u8);
                }
            }
        }
        w
    }

    fn test_acts(batch: usize, in_dim: usize) -> Vec<QuantizedActivation> {
        (0..batch)
            .map(|b| {
                let x: Vec<f32> = (0..in_dim)
                    .map(|i| ((b * 13 + i * 5) % 17) as f32 * 0.25 - 2.0)
                    .collect();
                let mut a = QuantizedActivation::with_capacity(in_dim);
                a.quantize(&x);
                a
            })
            .collect()
    }

    #[test]
    fn par_matmul_qdot_matches_matvec_per_batch_row() {
        let (out_dim, in_dim, batch) = (11, 64, 5);
        let w = q8_0_weights(out_dim, in_dim);
        let acts = test_acts(batch, in_dim);
        let pool = ThreadPool::new(4);

        let y_stride = out_dim + 3; // deliberately padded
        let mut got = vec![f32::NAN; batch * y_stride];
        par_matmul_qdot(
            &pool,
            QuantFormat::Q8_0,
            &w,
            &acts,
            &mut got,
            y_stride,
            0,
            out_dim,
            in_dim,
            SimdStrategy::Scalar,
        );

        for (b, act) in acts.iter().enumerate() {
            let mut want = vec![0f32; out_dim];
            par_matvec_qdot(
                &pool,
                QuantFormat::Q8_0,
                &w,
                act,
                &mut want,
                out_dim,
                in_dim,
                SimdStrategy::Scalar,
            );
            for o in 0..out_dim {
                let g = got[b * y_stride + o];
                assert!(
                    (g - want[o]).abs() < 1e-4,
                    "b={b} o={o}: got {g}, want {}",
                    want[o]
                );
            }
        }
    }

    #[test]
    fn par_matmul_qdot_x4_path_matches_single_dots_on_detected_backend() {
        // batch 6 = one x4 group + a 2-row remainder; the detected strategy
        // exercises the wide x4 kernel on AVX2/VNNI machines.
        let (out_dim, in_dim, batch) = (7, 96, 6);
        let w = q8_0_weights(out_dim, in_dim);
        let acts = test_acts(batch, in_dim);
        let pool = ThreadPool::new(4);
        let strategy = SimdStrategy::detect();

        let mut got = vec![f32::NAN; batch * out_dim];
        par_matmul_qdot(
            &pool,
            QuantFormat::Q8_0,
            &w,
            &acts,
            &mut got,
            out_dim,
            0,
            out_dim,
            in_dim,
            strategy,
        );

        for (b, act) in acts.iter().enumerate() {
            for o in 0..out_dim {
                let row = &w[o * (in_dim / 32 * 34)..(o + 1) * (in_dim / 32 * 34)];
                let want = row_dot_q8(QuantFormat::Q8_0, row, act, strategy);
                let g = got[b * out_dim + o];
                // x4 uses one accumulator chain per activation vs the
                // single dot's two — tiny f32 ordering differences allowed.
                let tol = want.abs().max(1.0) * 1e-5;
                assert!((g - want).abs() < tol, "b={b} o={o}: got {g}, want {want}");
            }
        }
    }

    #[test]
    fn par_matmul_qdot_col_off_places_columns() {
        let (out_dim, in_dim, batch) = (4, 32, 2);
        let w = q8_0_weights(out_dim, in_dim);
        let acts = test_acts(batch, in_dim);
        let pool = ThreadPool::new(2);

        let y_stride = 10;
        let col_off = 5;
        let mut y = vec![f32::NAN; batch * y_stride];
        par_matmul_qdot(
            &pool,
            QuantFormat::Q8_0,
            &w,
            &acts,
            &mut y,
            y_stride,
            col_off,
            out_dim,
            in_dim,
            SimdStrategy::Scalar,
        );
        for b in 0..batch {
            // Untouched columns stay NaN; written columns are finite.
            for c in 0..y_stride {
                let v = y[b * y_stride + c];
                if (col_off..col_off + out_dim).contains(&c) {
                    assert!(v.is_finite(), "b={b} c={c} should be written");
                } else {
                    assert!(v.is_nan(), "b={b} c={c} should be untouched");
                }
            }
        }
    }

    #[test]
    fn par_matmul_swiglu_matches_matvec_per_batch_row() {
        let (out_dim, in_dim, batch) = (9, 64, 4);
        let packed = q8_0_weights(out_dim * 2, in_dim); // gate/up interleaved
        let acts = test_acts(batch, in_dim);
        let pool = ThreadPool::new(4);

        let mut got = vec![f32::NAN; batch * out_dim];
        par_matmul_swiglu(
            &pool,
            QuantFormat::Q8_0,
            &packed,
            &acts,
            &mut got,
            out_dim,
            out_dim,
            in_dim,
            SimdStrategy::Scalar,
        );

        for (b, act) in acts.iter().enumerate() {
            let mut want = vec![0f32; out_dim];
            par_matvec_swiglu(
                &pool,
                QuantFormat::Q8_0,
                &packed,
                act,
                &mut want,
                out_dim,
                in_dim,
                SimdStrategy::Scalar,
            );
            for o in 0..out_dim {
                let g = got[b * out_dim + o];
                assert!(
                    (g - want[o]).abs() < 1e-4,
                    "b={b} o={o}: got {g}, want {}",
                    want[o]
                );
            }
        }
    }

    #[test]
    fn par_matmul_f32_matches_matvec_per_batch_row() {
        let (out_dim, in_dim, batch) = (13, 24, 3);
        let w: Vec<f32> = (0..out_dim * in_dim).map(|i| (i % 13) as f32 - 6.0).collect();
        let xb: Vec<f32> = (0..batch * in_dim)
            .map(|i| (i % 7) as f32 * 0.5 - 1.0)
            .collect();
        let pool = ThreadPool::new(4);

        let mut got = vec![f32::NAN; batch * out_dim];
        par_matmul(
            &pool,
            &w,
            &xb,
            in_dim,
            &mut got,
            out_dim,
            0,
            out_dim,
            in_dim,
            batch,
            SimdStrategy::Scalar,
        );

        for b in 0..batch {
            let mut want = vec![0f32; out_dim];
            par_matvec(
                &pool,
                &w,
                &xb[b * in_dim..(b + 1) * in_dim],
                &mut want,
                out_dim,
                in_dim,
                SimdStrategy::Scalar,
            );
            for o in 0..out_dim {
                assert!(
                    (got[b * out_dim + o] - want[o]).abs() < 1e-5,
                    "b={b} o={o}"
                );
            }
        }
    }

    #[test]
    fn par_matvec_matches_scalar() {
        let out_dim = 37; // deliberately not a multiple of thread count
        let in_dim = 24;
        let w: Vec<f32> = (0..out_dim * in_dim).map(|i| (i % 13) as f32 - 6.0).collect();
        let x: Vec<f32> = (0..in_dim).map(|i| (i % 7) as f32 * 0.5 - 1.0).collect();

        let mut want = vec![0f32; out_dim];
        crate::kernels::matmul::scalar::run_matvec(&w, &x, &mut want, out_dim, in_dim);

        let pool = ThreadPool::new(4);
        let mut got = vec![0f32; out_dim];
        par_matvec(&pool, &w, &x, &mut got, out_dim, in_dim, SimdStrategy::Scalar);

        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() < 1e-5, "got {g}, want {w}");
        }
    }
}
