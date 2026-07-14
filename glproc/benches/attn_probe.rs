//! Attention microbenchmark — isolate WHY the `attn` bucket costs 22x more per
//! MAC than the `qkv` bucket on the same machine.
//!
//! Measured on Qwen3-1.7B decode: attention runs at 0.83 GMAC/s while qkv runs
//! at 18.1 GMAC/s. Attention needs only 11.5 MMAC/token and reads 22.9 MB of KV
//! cache (against 1828 MB of weights), so by arithmetic it should be nearly
//! free. It is not.
//!
//! Two suspects, both visible in the source:
//!   A. the V-accumulation loop is scalar (attention.rs) while Q.K is AVX2
//!   B. the 16-head loop is single-threaded (runner.rs) — 3 cores idle
//!
//! This bench separates them so the fix is chosen from data, not from reading.
//! Run: cargo bench -p glproc --bench attn_probe

use std::time::Instant;

use glproc::attention::attention_one_into;
use glproc::simd_strategy::SimdStrategy;
use glproc::threading::ThreadPool;

/// Qwen3-1.7B decode shape.
const LAYERS: usize = 28;
const HEADS: usize = 16;
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
/// KvCache allocates for this, not for the live context. See the note in main().
const MAX_CONTEXT: usize = 4096;

fn prng(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
}

/// Current production path, verbatim: scalar V-accumulation, one head at a time.
fn baseline(
    q: &[f32],
    kc: &[f32],
    vc: &[f32],
    scores: &mut [f32],
    out: &mut [f32],
    cached: usize,
) {
    for h in 0..HEADS {
        let kv = h / (HEADS / KV_HEADS);
        let base = kv * MAX_CONTEXT * HEAD_DIM;
        let k = &kc[base..base + cached * HEAD_DIM];
        let v = &vc[base..base + cached * HEAD_DIM];
        attention_one_into(
            &q[h * HEAD_DIM..(h + 1) * HEAD_DIM],
            k,
            v,
            HEAD_DIM,
            scores,
            &mut out[h * HEAD_DIM..(h + 1) * HEAD_DIM],
        );
    }
}

/// Fix A only: AVX2 V-accumulation, still one thread.
///
/// `out[d] += w * v_row[d]` over head_dim=128 is 16 YMM lanes of pure FMA with
/// no reduction — the single easiest thing to vectorize in the whole engine,
/// and it is currently a scalar loop.
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn attn_head_simd(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    scores: &mut [f32],
    out: &mut [f32],
    cached: usize,
) {
    use std::arch::x86_64::*;

    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    for (t, s) in scores.iter_mut().enumerate().take(cached) {
        let k_row = &k_cache[t * HEAD_DIM..(t + 1) * HEAD_DIM];
        *s = glproc::kernels::matmul::avx2::dot_f32(q, k_row) * scale;
    }
    glproc::attention::softmax(&mut scores[..cached]);

    // Accumulate in registers: head_dim 128 = 16 YMM registers, which fits
    // (16 available). Zero, then FMA each cached row in. The scalar version
    // does exactly this one float at a time.
    for o in out.iter_mut() {
        *o = 0.0;
    }
    for (t, &w) in scores[..cached].iter().enumerate() {
        let wv = _mm256_set1_ps(w);
        let v_row = v_cache.as_ptr().add(t * HEAD_DIM);
        let mut d = 0;
        while d + 8 <= HEAD_DIM {
            let acc = _mm256_loadu_ps(out.as_ptr().add(d));
            let vv = _mm256_loadu_ps(v_row.add(d));
            _mm256_storeu_ps(out.as_mut_ptr().add(d), _mm256_fmadd_ps(wv, vv, acc));
            d += 8;
        }
    }
}

fn fix_a(q: &[f32], kc: &[f32], vc: &[f32], scores: &mut [f32], out: &mut [f32], cached: usize) {
    for h in 0..HEADS {
        let kv = h / (HEADS / KV_HEADS);
        let base = kv * MAX_CONTEXT * HEAD_DIM;
        let k = &kc[base..base + cached * HEAD_DIM];
        let v = &vc[base..base + cached * HEAD_DIM];
        // SAFETY: gated on SimdStrategy::detect() == Avx2 by the caller.
        unsafe {
            attn_head_simd(
                &q[h * HEAD_DIM..(h + 1) * HEAD_DIM],
                k,
                v,
                scores,
                &mut out[h * HEAD_DIM..(h + 1) * HEAD_DIM],
                cached,
            )
        };
    }
}

/// Raw pointer to the output, split across threads by head. Heads write disjoint
/// [h*head_dim, (h+1)*head_dim) ranges, so there is no race.
struct OutPtr(*mut f32);
unsafe impl Send for OutPtr {}
unsafe impl Sync for OutPtr {}

/// Fix A + B: AVX2 V-accumulation, heads split across the pool.
///
/// Each thread needs its own `scores` scratch (it is per-head state), so the
/// shared workspace buffer cannot be reused as-is — that is the one real
/// structural cost of threading this.
fn fix_ab(
    pool: &ThreadPool,
    q: &[f32],
    kc: &[f32],
    vc: &[f32],
    out: &mut [f32],
    cached: usize,
) {
    let outp = OutPtr(out.as_mut_ptr());
    let n = pool.n_threads();
    let chunk = HEADS.div_ceil(n);

    pool.run(&|tid| {
        let outp = &outp;
        // Per-thread scratch. A real implementation hoists this into the
        // Workspace (one scores buffer per thread) rather than allocating here.
        let mut scores = vec![0f32; cached];
        let lo = (tid * chunk).min(HEADS);
        let hi = (lo + chunk).min(HEADS);
        for h in lo..hi {
            let kv = h / (HEADS / KV_HEADS);
            let base = kv * MAX_CONTEXT * HEAD_DIM;
            let k = &kc[base..base + cached * HEAD_DIM];
            let v = &vc[base..base + cached * HEAD_DIM];
            // SAFETY: disjoint head ranges per thread; AVX2 checked by caller.
            let o = unsafe { std::slice::from_raw_parts_mut(outp.0.add(h * HEAD_DIM), HEAD_DIM) };
            unsafe {
                attn_head_simd(
                    &q[h * HEAD_DIM..(h + 1) * HEAD_DIM],
                    k,
                    v,
                    &mut scores,
                    o,
                    cached,
                )
            };
        }
    });
}

fn main() {
    let strategy = SimdStrategy::detect();
    if strategy != SimdStrategy::Avx2 {
        eprintln!("this probe assumes AVX2; detected {strategy:?} — results not meaningful");
    }
    let pool = ThreadPool::new(4);
    let mut seed = 0xA77Eu64;

    println!("Qwen3-1.7B attention probe — {LAYERS} layers x {HEADS} heads x {HEAD_DIM} dim");
    println!("simd {strategy:?} | pool {} threads\n", pool.n_threads());
    println!(
        "{:<8} {:>12} {:>12} {:>12} {:>10} {:>10}",
        "ctx", "baseline", "fix A simd", "fix A+B par", "A speedup", "AB speedup"
    );
    println!("{}", "-".repeat(70));

    for &cached in &[32usize, 64, 128, 256, 512, 1024] {
        let q: Vec<f32> = (0..HEADS * HEAD_DIM).map(|_| prng(&mut seed)).collect();

        // CRITICAL: reproduce the PRODUCTION KV layout, not a tight one.
        //
        // KvCache allocates for max_context (4096), so each head's region is
        // 4096 * 128 * 4 = 2 MB apart regardless of how much is actually
        // cached. Reading 16 heads therefore strides 2 MB between them — and
        // L2 is 1.25 MB, so every head starts cold. A tightly-packed benchmark
        // buffer (cached * head_dim) hides that entirely and measures a
        // workload the engine never runs.
        let stride = MAX_CONTEXT * HEAD_DIM; // floats between head regions
        let kc: Vec<f32> = (0..KV_HEADS * stride).map(|_| prng(&mut seed)).collect();
        let vc: Vec<f32> = (0..KV_HEADS * stride).map(|_| prng(&mut seed)).collect();
        let mut scores = vec![0f32; cached];
        let mut out = vec![0f32; HEADS * HEAD_DIM];

        // Correctness gate first: a faster wrong answer is worthless.
        baseline(&q, &kc, &vc, &mut scores, &mut out, cached);
        let want = out.clone();
        fix_a(&q, &kc, &vc, &mut scores, &mut out, cached);
        let worst_a = out
            .iter()
            .zip(&want)
            .map(|(g, w)| (g - w).abs() / w.abs().max(1.0))
            .fold(0f32, f32::max);
        fix_ab(&pool, &q, &kc, &vc, &mut out, cached);
        let worst_ab = out
            .iter()
            .zip(&want)
            .map(|(g, w)| (g - w).abs() / w.abs().max(1.0))
            .fold(0f32, f32::max);
        assert!(worst_a < 1e-5, "fix A diverged: {worst_a:.2e}");
        assert!(worst_ab < 1e-5, "fix A+B diverged: {worst_ab:.2e}");

        // Per call = one layer's attention for one token. Multiply by 28 layers
        // to compare against the profiler's per-token number.
        let iters = 2000;
        let time = |f: &mut dyn FnMut()| {
            for _ in 0..50 {
                f();
            }
            let t = Instant::now();
            for _ in 0..iters {
                f();
            }
            t.elapsed().as_secs_f64() * 1e6 / iters as f64 // microseconds
        };

        let base = time(&mut || baseline(&q, &kc, &vc, &mut scores, &mut out, cached));
        let a = time(&mut || fix_a(&q, &kc, &vc, &mut scores, &mut out, cached));
        let ab = time(&mut || fix_ab(&pool, &q, &kc, &vc, &mut out, cached));

        println!(
            "{cached:<8} {base:>10.1}us {a:>10.1}us {ab:>10.1}us {:>9.2}x {:>9.2}x",
            base / a,
            base / ab
        );
    }

    // ---------------------------------------------------------------------
    // The cold-cache run. The sweep above repeats ONE layer 2000x, so its KV
    // stays L2-hot and it under-reports by ~10x against the profiler (51.7us
    // vs 496us at ctx 128). Production walks 28 DIFFERENT layers per token,
    // each with its own multi-MB KV region, so every layer's KV is cold.
    //
    // Rotating over a full 28-layer cache reproduces that. This is the number
    // that should match the profiler, and the one any fix must actually beat.
    // ---------------------------------------------------------------------
    println!("\n--- cold cache: rotating over all {LAYERS} layers (production behavior) ---\n");

    // Measured from the real run, not assumed: GLPROC_ATTN_PROBE reports
    // "mean ctx 252" for glbench's default prompt (~220 prefill + 64 generated).
    // An earlier guess of 68 was wrong by 4x, which alone would have made every
    // number below meaningless.
    let cached = 252usize;
    let stride = MAX_CONTEXT * HEAD_DIM;
    let layer_span = KV_HEADS * stride;
    let q: Vec<f32> = (0..HEADS * HEAD_DIM).map(|_| prng(&mut seed)).collect();

    // One K and one V cache sized like the real thing: 28 layers x 8 kv heads
    // x 4096 ctx x 128 dim x 4 B = ~470 MB each. That is the actual footprint
    // the profiler reported as "kv cache 0.88 GiB".
    eprintln!(
        "allocating {:.2} GiB of KV (2 x {LAYERS} layers x {KV_HEADS} heads x {MAX_CONTEXT} ctx)...",
        2.0 * (LAYERS * layer_span * 4) as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    let kc: Vec<f32> = (0..LAYERS * layer_span).map(|_| prng(&mut seed)).collect();
    let vc: Vec<f32> = (0..LAYERS * layer_span).map(|_| prng(&mut seed)).collect();
    let mut scores = vec![0f32; cached];
    let mut out = vec![0f32; HEADS * HEAD_DIM];

    let iters = 200usize; // each iter = one full token: all 28 layers
    let time_tok = |f: &mut dyn FnMut(usize)| {
        for l in 0..LAYERS {
            f(l);
        }
        let t = Instant::now();
        for _ in 0..iters {
            for l in 0..LAYERS {
                f(l);
            }
        }
        t.elapsed().as_secs_f64() * 1e3 / iters as f64 // ms per TOKEN (28 layers)
    };

    let base_tok = time_tok(&mut |l| {
        let k = &kc[l * layer_span..(l + 1) * layer_span];
        let v = &vc[l * layer_span..(l + 1) * layer_span];
        baseline(&q, k, v, &mut scores, &mut out, cached);
    });
    let a_tok = time_tok(&mut |l| {
        let k = &kc[l * layer_span..(l + 1) * layer_span];
        let v = &vc[l * layer_span..(l + 1) * layer_span];
        fix_a(&q, k, v, &mut scores, &mut out, cached);
    });
    let ab_tok = time_tok(&mut |l| {
        let k = &kc[l * layer_span..(l + 1) * layer_span];
        let v = &vc[l * layer_span..(l + 1) * layer_span];
        fix_ab(&pool, &q, k, v, &mut out, cached);
    });

    println!("{:<24}{:>10}{:>11}{:>10}", "variant", "ms/token", "vs base", "us/layer");
    println!("{}", "-".repeat(56));
    for (name, t) in [
        ("baseline (current)", base_tok),
        ("fix A: simd V-accum", a_tok),
        ("fix A+B: + threaded", ab_tok),
    ] {
        println!(
            "{name:<24}{t:>10.2}{:>10.2}x{:>10.1}",
            base_tok / t,
            t * 1000.0 / LAYERS as f64
        );
    }

    println!("\nProfiler (real run): attn = 13.89 ms/token, 496 us/layer, 14.9% of decode.");
    println!("If 'baseline' above lands near 13.89 ms/token, the probe is faithful and");
    println!("the fix numbers are trustworthy. If it lands near 1.4 ms, the cache is");
    println!("still warm and something ELSE is costing 12 ms in the real attn bucket.");

    // ---------------------------------------------------------------------
    // Phase breakdown: WHERE inside attention does the time go?
    //
    // After SIMD (fix A) and threading (fix B), attention still runs at only
    // 40% of the bandwidth ceiling and 6.3 GMAC/s against the FFN's 22.9 —
    // 3.5x slower per MAC. Something inside the head loop is not accounted
    // for by "it reads the KV cache".
    //
    // Three timed variants, identical inputs, cold rotate over 28 layers:
    //   qk_only      = the Q.K dot loop, nothing else
    //   qk_softmax   = dots + softmax
    //   full         = attention_one_into (dots + softmax + V accumulation)
    // so softmax = (B - A) and V-accum = (C - B) by subtraction.
    //
    // The prime suspect going in: softmax calls the SCALAR fast_exp once per
    // cached position (252 per head, 16 heads, 28 layers = ~113k scalar exp
    // calls per token) while Q.K and V-accum are both vectorized. If softmax
    // dominates (B - A), the fix is a vectorized softmax, not a KV layout
    // change.
    // ---------------------------------------------------------------------
    println!("\n--- phase breakdown at ctx {cached}, cold rotate, sequential ---\n");

    let strategy2 = strategy; // captured by the closures below
    let qk_tok = time_tok(&mut |l| {
        let k = &kc[l * layer_span..(l + 1) * layer_span];
        for h in 0..HEADS {
            let kv = h / (HEADS / KV_HEADS);
            let base = kv * MAX_CONTEXT * HEAD_DIM;
            let krows = &k[base..base + cached * HEAD_DIM];
            let qh = &q[h * HEAD_DIM..(h + 1) * HEAD_DIM];
            let scale = 1.0 / (HEAD_DIM as f32).sqrt();
            for (t, s) in scores.iter_mut().enumerate().take(cached) {
                let row = &krows[t * HEAD_DIM..(t + 1) * HEAD_DIM];
                // SAFETY: strategy from detect(); AVX2 present.
                let dot = match strategy2 {
                    SimdStrategy::Avx512 | SimdStrategy::Avx2 => unsafe {
                        glproc::kernels::matmul::avx2::dot_f32(qh, row)
                    },
                    SimdStrategy::Scalar => glproc::kernels::matmul::scalar::dot_f32(qh, row),
                };
                *s = dot * scale;
            }
            std::hint::black_box(&mut scores);
        }
    });
    let qk_sm_tok = time_tok(&mut |l| {
        let k = &kc[l * layer_span..(l + 1) * layer_span];
        for h in 0..HEADS {
            let kv = h / (HEADS / KV_HEADS);
            let base = kv * MAX_CONTEXT * HEAD_DIM;
            let krows = &k[base..base + cached * HEAD_DIM];
            let qh = &q[h * HEAD_DIM..(h + 1) * HEAD_DIM];
            let scale = 1.0 / (HEAD_DIM as f32).sqrt();
            for (t, s) in scores.iter_mut().enumerate().take(cached) {
                let row = &krows[t * HEAD_DIM..(t + 1) * HEAD_DIM];
                let dot = match strategy2 {
                    SimdStrategy::Avx512 | SimdStrategy::Avx2 => unsafe {
                        glproc::kernels::matmul::avx2::dot_f32(qh, row)
                    },
                    SimdStrategy::Scalar => glproc::kernels::matmul::scalar::dot_f32(qh, row),
                };
                *s = dot * scale;
            }
            glproc::attention::softmax(&mut scores[..cached]);
            std::hint::black_box(&mut scores);
        }
    });
    let full_tok = time_tok(&mut |l| {
        let k = &kc[l * layer_span..(l + 1) * layer_span];
        let v = &vc[l * layer_span..(l + 1) * layer_span];
        baseline(&q, k, v, &mut scores, &mut out, cached);
    });

    println!("{:<18}{:>12}{:>12}", "phase", "ms/token", "share");
    println!("{}", "-".repeat(42));
    let sm = (qk_sm_tok - qk_tok).max(0.0);
    let vacc = (full_tok - qk_sm_tok).max(0.0);
    for (name, t) in [
        ("qk dots", qk_tok),
        ("softmax", sm),
        ("v-accum", vacc),
        ("TOTAL", full_tok),
    ] {
        println!("{name:<18}{t:>12.2}{:>11.0}%", t / full_tok * 100.0);
    }

    // ---------------------------------------------------------------------
    // Crossover sweep: at what context does threading start paying?
    //
    // ATTN_PAR_MIN_WORK was set to 1<<17 without measurement (parallel from
    // ctx >= 64 on this shape). This sweeps seq vs threaded on the cold rotate
    // so the threshold can be a measurement instead of a guess.
    // ---------------------------------------------------------------------
    println!("\n--- seq vs threaded crossover (cold rotate) ---\n");
    println!("{:<8}{:>12}{:>12}{:>10}", "ctx", "seq ms/tok", "par ms/tok", "speedup");
    println!("{}", "-".repeat(44));
    for &cx in &[16usize, 32, 64, 128, 256, 512, 1024] {
        let mut sc = vec![0f32; cx];
        let seq = time_tok(&mut |l| {
            let k = &kc[l * layer_span..(l + 1) * layer_span];
            let v = &vc[l * layer_span..(l + 1) * layer_span];
            baseline(&q, k, v, &mut sc, &mut out, cx);
        });
        let par = time_tok(&mut |l| {
            let k = &kc[l * layer_span..(l + 1) * layer_span];
            let v = &vc[l * layer_span..(l + 1) * layer_span];
            fix_ab(&pool, &q, k, v, &mut out, cx);
        });
        let mark = if par < seq { "  <- par wins" } else { "" };
        println!("{cx:<8}{seq:>12.3}{par:>12.3}{:>9.2}x{mark}", seq / par);
    }
}
