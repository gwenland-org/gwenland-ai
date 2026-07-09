//! Microbenchmark: isolate where decode time actually goes on real hardware.
//!
//! The T4 run showed decode stuck at ~105 tok/s (~9.5 ms/token) both before
//! and after attention fusion, and nvidia-smi reporting ~22% util — i.e. the
//! GPU is idle most of the time, so the bottleneck is latency/serialization,
//! not throughput, and it is NOT per-launch host overhead (fusion removed
//! ~900 launches/token with no effect). This bench measures the two prime
//! suspects directly, at Qwen2.5-0.5B decode dimensions:
//!
//!   1. Raw achievable memory bandwidth (streaming GB through a kernel).
//!   2. GEMV throughput at the real matvec sizes, timed over many iters with
//!      one sync at the end (amortizing launch cost) vs one sync per call
//!      (exposing per-call stall) — the gap is the serialization cost.
//!   3. A synchronous cuMemcpyDtoD loop at KV-write size (96/token) — tests
//!      whether the per-copy KV cache writes are the stall.
//!
//! Run: cargo run --release -p glcuda --example bench

use std::time::Instant;

use glcuda::buffer::BackendBuffer;
use glcuda::driver::{cuda_available, Cuda};
use glcuda::kernels::KernelSet;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("no CUDA device — nothing to benchmark");
        return Ok(());
    }
    let cuda = Cuda::probe()?;
    let k = KernelSet::load(&cuda)?;
    println!("device: {} sm_{}{}\n", cuda.info.name, cuda.info.sm_major, cuda.info.sm_minor);

    // --- Qwen2.5-0.5B decode dimensions ---
    let dim = 896usize;
    let hidden = 4864usize;
    let vocab = 151936usize;

    // A generously sized scratch buffer.
    let mut buf = BackendBuffer::new(&cuda, 1u64 << 30)?; // 1 GiB

    // ============================================================
    // 1. Achievable bandwidth: gl_add over a big buffer streams 2 reads +
    //    1 write per element. Time it, divide bytes moved by seconds.
    // ============================================================
    {
        let n = 64 * 1024 * 1024usize; // 64M f32 = 256 MB per array
        let a = buf.alloc_f32(n)?.dptr;
        let b = buf.alloc_f32(n)?.dptr;
        // warm
        k.add(&cuda, a, b, n as u32)?;
        cuda.synchronize()?;
        let iters = 20;
        let t = Instant::now();
        for _ in 0..iters {
            k.add(&cuda, a, b, n as u32)?; // reads a,b writes a: 3 * 4n bytes
        }
        cuda.synchronize()?;
        let secs = t.elapsed().as_secs_f64();
        let bytes = 3.0 * 4.0 * n as f64 * iters as f64;
        println!(
            "[bandwidth] gl_add {n} elems x{iters}: {:.2} ms total, {:.0} GB/s achievable",
            secs * 1e3,
            bytes / secs / 1e9
        );
        buf.reset_to(0);
    }

    // ============================================================
    // 2. GEMV throughput at real decode sizes. Two timings:
    //    (a) N launches then ONE sync  -> pure GPU throughput (best case)
    //    (b) N launches each with a sync -> per-call latency exposed
    // ============================================================
    for (label, out_dim, in_dim) in
        [("qkv   ", dim + 2 * 128, dim), ("gate  ", hidden, dim), ("down  ", dim, hidden), ("lmhead", vocab, dim)]
    {
        let mark = buf.mark();
        let w = buf.alloc_f32(out_dim * in_dim)?.dptr;
        let x = buf.alloc_f32(in_dim)?.dptr;
        let y = buf.alloc_f32(out_dim)?.dptr;
        k.gemv(&cuda, w, x, y, out_dim as u32, in_dim as u32)?;
        cuda.synchronize()?;
        let iters = 200;

        // (a) batched: all launches, one sync
        let t = Instant::now();
        for _ in 0..iters {
            k.gemv(&cuda, w, x, y, out_dim as u32, in_dim as u32)?;
        }
        cuda.synchronize()?;
        let batched = t.elapsed().as_secs_f64() / iters as f64;

        // (b) per-call sync
        let t = Instant::now();
        for _ in 0..iters {
            k.gemv(&cuda, w, x, y, out_dim as u32, in_dim as u32)?;
            cuda.synchronize()?;
        }
        let synced = t.elapsed().as_secs_f64() / iters as f64;

        let bytes = 4.0 * (out_dim * in_dim) as f64; // weight stream dominates
        println!(
            "[gemv {label}] {out_dim:>6}x{in_dim:<5}  batched {:>6.1} us ({:>5.0} GB/s) | per-sync {:>6.1} us  | stall {:>5.1} us",
            batched * 1e6,
            bytes / batched / 1e9,
            synced * 1e6,
            (synced - batched) * 1e6,
        );
        buf.reset_to(mark);
    }

    // ============================================================
    // 2b. Q8_0 GEMV — the kernel the MODEL actually runs (gemv_q8_0), at the
    //     same decode sizes. This is the real decode path; the f32 gemv above
    //     is not. Q8_0 streams ~1.06 B/weight vs 4 B, so compare its GB/s to
    //     the 265 achievable to see if the quantized kernel is the bottleneck.
    // ============================================================
    for (label, out_dim, in_dim) in
        [("gate  ", 2 * hidden, dim), ("down  ", dim, hidden), ("lmhead", vocab, dim)]
    {
        let mark = buf.mark();
        // Q8_0: 34 bytes per 32-weight block.
        let row_blocks = in_dim / 32;
        let wbytes = out_dim * row_blocks * 34;
        let w = buf.alloc(wbytes as u64)?.dptr;
        let x = buf.alloc_f32(in_dim)?.dptr;
        let y = buf.alloc_f32(out_dim)?.dptr;
        k.gemv_q8_0(&cuda, w, x, y, out_dim as u32, in_dim as u32)?;
        cuda.synchronize()?;
        let iters = 200;
        let t = Instant::now();
        for _ in 0..iters {
            k.gemv_q8_0(&cuda, w, x, y, out_dim as u32, in_dim as u32)?;
        }
        cuda.synchronize()?;
        let batched = t.elapsed().as_secs_f64() / iters as f64;
        println!(
            "[q8_0 {label}] {out_dim:>6}x{in_dim:<5}  batched {:>6.1} us ({:>5.0} GB/s of {wbytes} B)",
            batched * 1e6,
            wbytes as f64 / batched / 1e9,
        );
        buf.reset_to(mark);
    }

    // ============================================================
    // 3. Synchronous DtoD at KV-write granularity (head_dim=64 f32 = 256 B),
    //    96 copies per token — is the per-copy latency the stall?
    // ============================================================
    {
        let head_bytes = 64 * 4usize;
        let src = buf.alloc_f32(64)?.dptr;
        let dst = buf.alloc_f32(64 * 4096)?.dptr;
        cuda.dtod(dst, src, head_bytes)?;
        cuda.synchronize()?;
        let tokens = 200;
        let copies_per_token = 96;
        let t = Instant::now();
        for _ in 0..tokens {
            for i in 0..copies_per_token {
                cuda.dtod(dst + (i * head_bytes) as u64, src, head_bytes)?;
            }
        }
        cuda.synchronize()?;
        let per_token = t.elapsed().as_secs_f64() / tokens as f64;
        println!(
            "\n[kv dtod] {copies_per_token} copies/token x{tokens}: {:.2} ms/token ({:.1} us/copy)",
            per_token * 1e3,
            per_token / copies_per_token as f64 * 1e6,
        );
    }

    buf.free(&cuda)?;
    println!("\ndone.");
    Ok(())
}
