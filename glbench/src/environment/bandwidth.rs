//! Measured memory read bandwidth — the ceiling every other number is judged
//! against.
//!
//! # Why this exists
//!
//! glbench had a full roofline analysis (`analysis::ceiling`, `analysis::
//! bottleneck`) that never fired on CPU runs, because it only ever looked for a
//! *GPU's* published peak bandwidth. Every CPU session therefore reported
//! `bottleneck: undetermined` — the machinery was there, the number was not.
//!
//! Without a ceiling, "23.0 GB/s" is not a fact anyone can act on. It could be
//! 78% of the machine (nothing left to win) or 30% (something is badly wrong),
//! and those call for opposite decisions. Measuring the ceiling turns every
//! other bandwidth figure into a fraction.
//!
//! # Method
//!
//! Stream a buffer several times larger than L3 and time the reads. Nothing
//! clever: a sequential sum over a buffer big enough that it cannot be cached is
//! exactly the access pattern decode has (weights streamed once per token, no
//! reuse), so it measures the ceiling *for the workload we care about* rather
//! than a vendor's theoretical peak.
//!
//! Deliberately **not** a vendor lookup table: DDR4-2667 dual-channel and
//! DDR4-2667 single-channel differ by 2x, and no CPUID bit tells them apart. A
//! published figure would describe a machine we might not be sitting on.

use std::time::Instant;

/// Buffer size. Must comfortably exceed L3 (6 MB on the Tiger Lake baseline, up
/// to ~64 MB on server parts) so every read misses cache and goes to DRAM.
const BUF_BYTES: usize = 256 * 1024 * 1024;

/// Passes to time. The first is discarded — it faults the pages in, which
/// measures the page allocator rather than the memory bus.
const PASSES: usize = 4;

/// Measure sustained sequential read bandwidth, GB/s.
///
/// **Multi-threaded on purpose.** A single core cannot saturate a modern memory
/// bus — it runs out of outstanding-miss slots (line-fill buffers) long before
/// the DRAM controller runs out of throughput. A single-threaded probe therefore
/// measures the *per-core* ceiling, and using that as the machine ceiling makes
/// every engine stage look like it exceeds 100% of it (measured: it reported
/// 17.6 GB/s where threaded reads reach ~29, so a 4-thread engine appeared to
/// run at "146% of peak" — an obviously impossible number that was the tell).
///
/// The engine reads with `n_threads` workers, so the ceiling must be measured
/// the same way, or the comparison is between two different machines.
///
/// Returns `None` if allocation fails or the timing is degenerate, rather than
/// reporting a fabricated ceiling — a wrong ceiling is worse than none, because
/// every efficiency figure downstream inherits its error.
pub fn measure_read_gbs() -> Option<f64> {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    // u64 elements: 8 bytes per read keeps the loop's own overhead well below
    // the memory latency it is trying to measure.
    let n = BUF_BYTES / std::mem::size_of::<u64>();
    let mut buf: Vec<u64> = Vec::new();
    buf.try_reserve_exact(n).ok()?;
    // Non-zero, varying content: a page of zeros can be backed by a single
    // shared physical page (or compressed), which would measure nothing.
    buf.extend((0..n).map(|i| (i as u64).wrapping_mul(0x9E3779B97F4A7C15)));
    let buf = &buf[..];

    // Warm-up: fault every page in. Timing this would measure demand-zero page
    // faults, not bandwidth.
    let _ = sum_pass(buf);

    let chunk = n.div_ceil(threads);
    let mut best = 0f64;
    for _ in 0..PASSES {
        let t = Instant::now();
        // Contiguous chunk per thread — the same access shape the engine's
        // matvec kernels use, and the one single-channel DDR4 rewards.
        std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(threads);
            for w in 0..threads {
                let lo = (w * chunk).min(n);
                let hi = (lo + chunk).min(n);
                handles.push(s.spawn(move || sum_pass(&buf[lo..hi])));
            }
            let mut acc = 0u64;
            for h in handles {
                acc = acc.wrapping_add(h.join().unwrap_or(0));
            }
            std::hint::black_box(acc);
        });
        let el = t.elapsed().as_secs_f64();
        if el > 0.0 {
            // Take the BEST pass, not the mean. A slow pass means the OS
            // scheduled something else on our cores or the part throttled; the
            // ceiling is what the machine *can* do, not what it averaged while
            // being interrupted.
            best = best.max(BUF_BYTES as f64 / el / 1e9);
        }
    }
    (best > 0.0).then_some(best)
}

/// Sequential sum. Four independent accumulators so the loop is not serialized
/// on the add's latency — otherwise this measures ALU dependency chains rather
/// than the memory bus.
fn sum_pass(buf: &[u64]) -> u64 {
    let mut a = [0u64; 4];
    let mut i = 0;
    while i + 4 <= buf.len() {
        a[0] = a[0].wrapping_add(buf[i]);
        a[1] = a[1].wrapping_add(buf[i + 1]);
        a[2] = a[2].wrapping_add(buf[i + 2]);
        a[3] = a[3].wrapping_add(buf[i + 3]);
        i += 4;
    }
    for &v in &buf[i..] {
        a[0] = a[0].wrapping_add(v);
    }
    a[0]
        .wrapping_add(a[1])
        .wrapping_add(a[2])
        .wrapping_add(a[3])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measures_a_plausible_ceiling() {
        let Some(gbs) = measure_read_gbs() else {
            // Allocation can fail on a constrained runner; that is a valid
            // outcome, not a test failure.
            return;
        };
        // Sanity band, not a target: no DDR system reads at under 1 GB/s, and
        // none at over 2 TB/s. A number outside this means the measurement is
        // broken (compiler elided the loop, timer resolution, etc.) and a broken
        // ceiling silently corrupts every efficiency figure downstream.
        assert!(
            (1.0..2000.0).contains(&gbs),
            "implausible bandwidth {gbs:.1} GB/s — measurement is broken"
        );
        eprintln!("measured read bandwidth: {gbs:.1} GB/s");
    }

    /// The ceiling must be measured with the same parallelism the engine uses.
    ///
    /// A single core cannot saturate a memory bus, so a single-threaded probe
    /// measures the *per-core* ceiling. The first version of `measure_read_gbs`
    /// did exactly that — reported 17.6 GB/s, and the 4-thread engine then
    /// appeared to run at **146% of peak**. An impossible number was the tell,
    /// and this test exists so it cannot come back.
    ///
    /// Both measurements are taken **inside this one test**, back to back. An
    /// earlier version called `measure_read_gbs()` and compared against a
    /// separately-timed single-core pass — and failed intermittently, because
    /// the test harness runs tests in parallel and a *different* test saturating
    /// the bus made the threaded number collapse. A bandwidth probe cannot be
    /// honest while something else is competing for the bus, so the comparison
    /// has to happen under identical conditions or not at all.
    #[test]
    fn threaded_read_beats_single_core() {
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        if threads < 2 {
            // On a single-core machine the two are the same measurement.
            return;
        }

        let n = BUF_BYTES / std::mem::size_of::<u64>();
        let mut buf: Vec<u64> = Vec::new();
        if buf.try_reserve_exact(n).is_err() {
            return; // constrained runner; not a failure
        }
        buf.extend((0..n).map(|i| (i as u64).wrapping_mul(0x9E3779B97F4A7C15)));
        let buf = &buf[..];
        let _ = sum_pass(buf); // fault the pages in

        // Best-of-N each, interleaved, so a transient stall on the machine hits
        // both arms rather than only one.
        let mut single = 0f64;
        let mut multi = 0f64;
        let chunk = n.div_ceil(threads);
        for _ in 0..PASSES {
            let t = Instant::now();
            std::hint::black_box(sum_pass(buf));
            single = single.max(BUF_BYTES as f64 / t.elapsed().as_secs_f64() / 1e9);

            let t = Instant::now();
            std::thread::scope(|s| {
                let hs: Vec<_> = (0..threads)
                    .map(|w| {
                        let lo = (w * chunk).min(n);
                        let hi = (lo + chunk).min(n);
                        s.spawn(move || sum_pass(&buf[lo..hi]))
                    })
                    .collect();
                let mut acc = 0u64;
                for h in hs {
                    acc = acc.wrapping_add(h.join().unwrap_or(0));
                }
                std::hint::black_box(acc);
            });
            multi = multi.max(BUF_BYTES as f64 / t.elapsed().as_secs_f64() / 1e9);
        }

        eprintln!("bandwidth: {single:.1} GB/s single-core, {multi:.1} GB/s threaded");
        // Threading must not make reads *slower*. It should make them faster,
        // but the load-bearing claim is only that the ceiling we report is not
        // below what one core already achieves — that is the state that produced
        // the impossible ">100% of peak" figures.
        assert!(
            multi >= single * 0.95,
            "threaded read {multi:.1} GB/s is below single-core {single:.1} — \
             the probe would report a ceiling the engine can exceed"
        );
    }

    #[test]
    fn sum_pass_reads_every_element() {
        // Guards against an off-by-one that would skip the tail and silently
        // overstate bandwidth by reading less than we claim.
        let buf: Vec<u64> = (1..=10).collect();
        assert_eq!(sum_pass(&buf), 55, "must sum all 10 elements");
        let buf: Vec<u64> = vec![1; 7]; // not a multiple of 4
        assert_eq!(sum_pass(&buf), 7);
    }
}
