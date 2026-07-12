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
    //     is not. The dp4a path takes int8-quantized activations + per-32
    //     scales, so we quantize x once (as the runner does per matmul) then
    //     time ONLY the GEMV — matching what an nsys per-kernel node reports.
    //     Weights are PADDED to 36 B per 32-weight block (the kernel's row
    //     stride is n_blocks * 36); compare GB/s to the ~265 achievable to see
    //     if the quantized kernel is bandwidth-bound.
    // ============================================================
    for (label, out_dim, in_dim) in
        [("gate  ", 2 * hidden, dim), ("down  ", dim, hidden), ("lmhead", vocab, dim)]
    {
        let mark = buf.mark();
        let row_blocks = in_dim / 32;
        let wbytes = out_dim * row_blocks * 36;
        let w = buf.alloc(wbytes as u64)?.dptr;
        // dp4a activation operands: int8 qs (1 B/elem) + one f32 scale/block.
        let x = buf.alloc_f32(in_dim)?.dptr;
        let x_qs = buf.alloc(in_dim as u64)?.dptr;
        let x_scales = buf.alloc_f32(in_dim / 32)?.dptr;
        let y = buf.alloc_f32(out_dim)?.dptr;
        k.quantize_q8(&cuda, x, x_qs, x_scales, in_dim as u32)?;
        k.gemv_q8_0(&cuda, w, x_qs, x_scales, y, out_dim as u32, in_dim as u32)?;
        cuda.synchronize()?;
        let iters = 200;
        let t = Instant::now();
        for _ in 0..iters {
            k.gemv_q8_0(&cuda, w, x_qs, x_scales, y, out_dim as u32, in_dim as u32)?;
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
    // 2d. Q8_0 SoA GEMV — new contiguous-qs kernel. Validates correctness vs a
    //     CPU reference (the correct oracle) and times it vs the AoS kernel to
    //     see the bandwidth gain from killing the interleaved-scale/padding
    //     coalescing loss. Useful stream = out*in qs + out*nblocks*2 scale
    //     bytes (no 36 B padding), so GB/s is on real bytes moved.
    // ============================================================
    {
        // Deterministic pseudo-data (no host f16 conversion: we pick f16 scale
        // bit patterns whose exact f32 value we also know for the CPU ref).
        let scale_bits: [u16; 4] = [0x3C00, 0x3800, 0x3E00, 0x3400]; // 1.0 0.5 1.5 0.25
        let scale_vals: [f32; 4] = [1.0, 0.5, 1.5, 0.25];
        let qb = |i: usize| -> i8 { (((i * 131 + 7) % 255) as i32 - 127) as i8 };

        for (label, out_dim, in_dim) in
            [("gate  ", 2 * hidden, dim), ("down  ", dim, hidden), ("lmhead", vocab, dim)]
        {
            let mark = buf.mark();
            let nb = in_dim / 32;
            // host weights: qs (int8) + per-block f16 scales.
            let wqs: Vec<i8> = (0..out_dim * in_dim).map(qb).collect();
            let wsc_bits: Vec<u16> = (0..out_dim * nb).map(|i| scale_bits[i % 4]).collect();
            let xqs: Vec<i8> = (0..in_dim).map(|i| qb(i * 7 + 3)).collect();
            let xsc: Vec<f32> = (0..nb).map(|b| 0.5 + (b % 3) as f32 * 0.25).collect();

            // device SoA buffers.
            let d_wqs = buf.alloc((out_dim * in_dim) as u64)?.dptr;
            let d_wsc = buf.alloc((out_dim * nb * 2) as u64)?.dptr;
            let d_xqs = buf.alloc(in_dim as u64)?.dptr;
            let d_xsc = buf.alloc_f32(nb)?.dptr;
            let d_y = buf.alloc_f32(out_dim)?.dptr;
            let as_u8 = |v: &[i8]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len()) };
            let wsc_u8 =
                unsafe { std::slice::from_raw_parts(wsc_bits.as_ptr() as *const u8, wsc_bits.len() * 2) };
            cuda.htod(d_wqs, as_u8(&wqs))?;
            cuda.htod(d_wsc, wsc_u8)?;
            cuda.htod(d_xqs, as_u8(&xqs))?;
            cuda.htod_f32(d_xsc, &xsc)?;

            k.gemv_q8_0_soa(&cuda, d_wqs, d_wsc, d_xqs, d_xsc, d_y, out_dim as u32, in_dim as u32)?;
            cuda.synchronize()?;

            // CPU reference: y[r] = sum_b (wsc*xsc) * sum_j wqs*xqs.
            let mut y_host = vec![0f32; out_dim];
            cuda.dtoh_f32(&mut y_host, d_y)?;
            let mut max_rel = 0f32;
            for r in 0..out_dim {
                let mut acc = 0f32;
                for b in 0..nb {
                    let mut dot = 0i32;
                    for j in 0..32 {
                        dot += wqs[r * in_dim + b * 32 + j] as i32 * xqs[b * 32 + j] as i32;
                    }
                    acc += dot as f32 * (scale_vals[(r * nb + b) % 4] * xsc[b]);
                }
                let d = (acc - y_host[r]).abs() / acc.abs().max(1.0);
                max_rel = max_rel.max(d);
            }

            let iters = 200;
            let t = Instant::now();
            for _ in 0..iters {
                k.gemv_q8_0_soa(&cuda, d_wqs, d_wsc, d_xqs, d_xsc, d_y, out_dim as u32, in_dim as u32)?;
            }
            cuda.synchronize()?;
            let soa = t.elapsed().as_secs_f64() / iters as f64;
            let sbytes = out_dim * in_dim + out_dim * nb * 2;
            println!(
                "[soa  {label}] {out_dim:>6}x{in_dim:<5}  batched {:>6.1} us ({:>5.0} GB/s of {sbytes} B) | max_rel_err {:.1e}",
                soa * 1e6,
                sbytes as f64 / soa / 1e9,
                max_rel,
            );
            buf.reset_to(mark);
        }
    }

    // ============================================================
    // 2g. Q4_K SoA GEMV (gl_gemv_q4_k_soa, M2.1 Task A) — the Q4_K_M decode
    //     kernel, at Qwen2.5-7B shapes (in_dim must be a multiple of 256, so
    //     the 0.5B dims don't apply). Validates vs a CPU reference on
    //     synthetic SoA streams, then reports GB/s of the real bytes
    //     streamed (qs + f16 scales + f16 mins = 5.0 bpw) against the
    //     achievable ceiling — isolating KERNEL efficiency from model-level
    //     byte inflation (the Q6_K->Q8_0 requant tensors are a separate,
    //     already-measured stream).
    // ============================================================
    {
        // f16 bit patterns with exact f32 values (same trick as 2d).
        let sc_bits: [u16; 4] = [0x3C00, 0x3800, 0x3E00, 0x3400]; // 1.0 0.5 1.5 0.25
        let sc_vals: [f32; 4] = [1.0, 0.5, 1.5, 0.25];
        let mn_bits: [u16; 4] = [0x3400, 0x2C00, 0x3800, 0x3000]; // 0.25 0.0625 0.5 0.125
        let mn_vals: [f32; 4] = [0.25, 0.0625, 0.5, 0.125];
        let qb = |i: usize| -> u8 { ((i * 131 + 7) % 251) as u8 };

        let (dim7, hidden7) = (3584usize, 18944usize); // Qwen2.5-7B
        for (label, out_dim, in_dim) in [("gate7b", 2 * hidden7, dim7), ("down7b", dim7, hidden7)] {
            let mark = buf.mark();
            let nsub = in_dim / 32; // f16 scale/min pairs per row

            // Host streams in the kernel's SoA layout.
            let wqs: Vec<u8> = (0..out_dim * in_dim / 2).map(qb).collect();
            let wsc_bits: Vec<u16> = (0..out_dim * nsub).map(|i| sc_bits[i % 4]).collect();
            let wmn_bits: Vec<u16> = (0..out_dim * nsub).map(|i| mn_bits[(i / 3) % 4]).collect();
            let xqs: Vec<i8> = (0..in_dim).map(|i| (qb(i * 7 + 3) as i32 - 125) as i8).collect();
            let xsc: Vec<f32> = (0..nsub).map(|b| 0.5 + (b % 3) as f32 * 0.25).collect();

            let d_wqs = buf.alloc(wqs.len() as u64)?.dptr;
            let d_wsc = buf.alloc((wsc_bits.len() * 2) as u64)?.dptr;
            let d_wmn = buf.alloc((wmn_bits.len() * 2) as u64)?.dptr;
            let d_xqs = buf.alloc(in_dim as u64)?.dptr;
            let d_xsc = buf.alloc_f32(nsub)?.dptr;
            let d_y = buf.alloc_f32(out_dim)?.dptr;
            let u16_u8 = |v: &[u16]| unsafe {
                std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2)
            };
            cuda.htod(d_wqs, &wqs)?;
            cuda.htod(d_wsc, u16_u8(&wsc_bits))?;
            cuda.htod(d_wmn, u16_u8(&wmn_bits))?;
            cuda.htod(d_xqs, unsafe {
                std::slice::from_raw_parts(xqs.as_ptr() as *const u8, xqs.len())
            })?;
            cuda.htod_f32(d_xsc, &xsc)?;

            k.gemv_q4_k_soa(&cuda, d_wqs, d_wsc, d_wmn, d_xqs, d_xsc, d_y, out_dim as u32, in_dim as u32)?;
            cuda.synchronize()?;
            let mut y_host = vec![0f32; out_dim];
            cuda.dtoh_f32(&mut y_host, d_y)?;

            // CPU reference straight off the SoA streams: per 32-value
            // sub-block, acc += (sc*xs)*dot(q,xq) - (mn*xs)*sum(xq), with the
            // kernel's nibble order (value v of a super-block: group g=v/8,
            // r=v%8 -> byte g*4 + r%4, low nibble when r<4).
            let mut max_rel = 0f32;
            for r in 0..out_dim {
                let mut acc = 0f32;
                for s in 0..nsub {
                    let sc = sc_vals[(r * nsub + s) % 4];
                    let mn = mn_vals[((r * nsub + s) / 3) % 4];
                    let xs = xsc[s];
                    let (mut dot, mut sum) = (0i32, 0i32);
                    for i in 0..32 {
                        let v = s * 32 + i; // linear index within the row
                        let (sb, local) = (v / 256, v % 256);
                        let (g, rr) = (local / 8, local % 8);
                        let byte = wqs[r * in_dim / 2 + sb * 128 + g * 4 + (rr % 4)];
                        let q = if rr < 4 { byte & 0x0F } else { byte >> 4 } as i32;
                        let x = xqs[v] as i32;
                        dot += q * x;
                        sum += x;
                    }
                    acc += sc * xs * dot as f32 - mn * xs * sum as f32;
                }
                let d = (acc - y_host[r]).abs() / acc.abs().max(1.0);
                max_rel = max_rel.max(d);
            }

            let iters = 100;
            let t = Instant::now();
            for _ in 0..iters {
                k.gemv_q4_k_soa(&cuda, d_wqs, d_wsc, d_wmn, d_xqs, d_xsc, d_y, out_dim as u32, in_dim as u32)?;
            }
            cuda.synchronize()?;
            let each = t.elapsed().as_secs_f64() / iters as f64;
            let sbytes = wqs.len() + wsc_bits.len() * 2 + wmn_bits.len() * 2;
            println!(
                "[q4k  {label}] {out_dim:>6}x{in_dim:<5}  batched {:>6.1} us ({:>5.0} GB/s of {sbytes} B) | max_rel_err {:.1e}",
                each * 1e6,
                sbytes as f64 / each / 1e9,
                max_rel,
            );
            buf.reset_to(mark);
        }
    }

    // ============================================================
    // 2h. Q4_0 SoA GEMV (gl_gemv_q4_0_soa, M2.2 Task C-2) — the 4.5 bpw
    //     decode kernel for Q4_0 files, at 7B shapes. Same method as [q4k]:
    //     CPU-validated on synthetic SoA streams, GB/s of real bytes moved
    //     (qs + verbatim f16 scales) vs the achievable ceiling.
    // ============================================================
    {
        let sc_bits: [u16; 4] = [0x3C00, 0x3800, 0x3E00, 0x3400]; // 1.0 0.5 1.5 0.25
        let sc_vals: [f32; 4] = [1.0, 0.5, 1.5, 0.25];
        let qb = |i: usize| -> u8 { ((i * 131 + 7) % 251) as u8 };

        let (dim7, hidden7) = (3584usize, 18944usize);
        for (label, out_dim, in_dim) in [("gate7b", 2 * hidden7, dim7), ("down7b", dim7, hidden7)] {
            let mark = buf.mark();
            let nb = in_dim / 32;

            let wqs: Vec<u8> = (0..out_dim * in_dim / 2).map(qb).collect();
            let wsc_bits: Vec<u16> = (0..out_dim * nb).map(|i| sc_bits[i % 4]).collect();
            let xqs: Vec<i8> = (0..in_dim).map(|i| (qb(i * 7 + 3) as i32 - 125) as i8).collect();
            let xsc: Vec<f32> = (0..nb).map(|b| 0.5 + (b % 3) as f32 * 0.25).collect();

            let d_wqs = buf.alloc(wqs.len() as u64)?.dptr;
            let d_wsc = buf.alloc((wsc_bits.len() * 2) as u64)?.dptr;
            let d_xqs = buf.alloc(in_dim as u64)?.dptr;
            let d_xsc = buf.alloc_f32(nb)?.dptr;
            let d_y = buf.alloc_f32(out_dim)?.dptr;
            let u16_u8 = |v: &[u16]| unsafe {
                std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2)
            };
            cuda.htod(d_wqs, &wqs)?;
            cuda.htod(d_wsc, u16_u8(&wsc_bits))?;
            cuda.htod(d_xqs, unsafe {
                std::slice::from_raw_parts(xqs.as_ptr() as *const u8, xqs.len())
            })?;
            cuda.htod_f32(d_xsc, &xsc)?;

            k.gemv_q4_0_soa(&cuda, d_wqs, d_wsc, d_xqs, d_xsc, d_y, out_dim as u32, in_dim as u32)?;
            cuda.synchronize()?;
            let mut y_host = vec![0f32; out_dim];
            cuda.dtoh_f32(&mut y_host, d_y)?;

            // CPU reference off the SoA streams: per 32-value block,
            // acc += d*xs*(dot(q,xq) - 8*sum(xq)), kernel nibble order.
            let mut max_rel = 0f32;
            for r in 0..out_dim {
                let mut acc = 0f32;
                for b in 0..nb {
                    let d = sc_vals[(r * nb + b) % 4];
                    let xs = xsc[b];
                    let (mut dot, mut sum) = (0i32, 0i32);
                    for i in 0..32 {
                        let v = b * 32 + i;
                        let (g, rr) = (v / 8, v % 8);
                        let byte = wqs[r * in_dim / 2 + g * 4 + (rr % 4)];
                        let q = if rr < 4 { byte & 0x0F } else { byte >> 4 } as i32;
                        let xv = xqs[v] as i32;
                        dot += q * xv;
                        sum += xv;
                    }
                    acc += d * xs * (dot - 8 * sum) as f32;
                }
                let dl = (acc - y_host[r]).abs() / acc.abs().max(1.0);
                max_rel = max_rel.max(dl);
            }

            let iters = 100;
            let t = Instant::now();
            for _ in 0..iters {
                k.gemv_q4_0_soa(&cuda, d_wqs, d_wsc, d_xqs, d_xsc, d_y, out_dim as u32, in_dim as u32)?;
            }
            cuda.synchronize()?;
            let each = t.elapsed().as_secs_f64() / iters as f64;
            let sbytes = wqs.len() + wsc_bits.len() * 2;
            println!(
                "[q4_0 {label}] {out_dim:>6}x{in_dim:<5}  batched {:>6.1} us ({:>5.0} GB/s of {sbytes} B) | max_rel_err {:.1e}",
                each * 1e6,
                sbytes as f64 / each / 1e9,
                max_rel,
            );
            buf.reset_to(mark);
        }
    }

    // ============================================================
    // 2i. Q6_K SoA GEMV (gl_gemv_q6_k_soa, M2.2 Task C-1) — 7.0625 bpw
    //     (qh widened for ALU, see repack) replacing the Q6_K->Q8_0 requant
    //     stream at 8.5. Same method: CPU-validated on synthetic SoA
    //     streams, GB/s of real bytes vs the achievable ceiling.
    // ============================================================
    {
        let d_bits: [u16; 4] = [0x3C00, 0x3800, 0x3E00, 0x3400]; // 1.0 0.5 1.5 0.25
        let d_vals: [f32; 4] = [1.0, 0.5, 1.5, 0.25];
        let qb = |i: usize| -> u8 { ((i * 131 + 7) % 251) as u8 };

        let (dim7, hidden7) = (3584usize, 18944usize);
        for (label, out_dim, in_dim) in [("gate7b", 2 * hidden7, dim7), ("down7b", dim7, hidden7)] {
            let mark = buf.mark();
            let nsb = in_dim / 256; // super-blocks per row
            let nsub = in_dim / 16; // i8 scales per row

            let wql: Vec<u8> = (0..out_dim * in_dim / 2).map(qb).collect();
            // qh: widened nibble layout, each nibble a 2-bit field (0..3).
            let wqh: Vec<u8> = (0..out_dim * in_dim / 2).map(|i| qb(i * 3 + 1) & 0x33).collect();
            let wsc: Vec<i8> = (0..out_dim * nsub).map(|i| ((i * 5) % 23) as i8 - 11).collect();
            let wd_bits: Vec<u16> = (0..out_dim * nsb).map(|i| d_bits[i % 4]).collect();
            let xqs: Vec<i8> = (0..in_dim).map(|i| (qb(i * 7 + 3) as i32 - 125) as i8).collect();
            let xsc: Vec<f32> = (0..in_dim / 32).map(|b| 0.5 + (b % 3) as f32 * 0.25).collect();

            let d_wql = buf.alloc(wql.len() as u64)?.dptr;
            let d_wqh = buf.alloc(wqh.len() as u64)?.dptr;
            let d_wsc = buf.alloc(wsc.len() as u64)?.dptr;
            let d_wd = buf.alloc((wd_bits.len() * 2) as u64)?.dptr;
            let d_xqs = buf.alloc(in_dim as u64)?.dptr;
            let d_xsc = buf.alloc_f32(in_dim / 32)?.dptr;
            let d_y = buf.alloc_f32(out_dim)?.dptr;
            let as_u8 = |v: &[i8]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len()) };
            let u16_u8 = |v: &[u16]| unsafe {
                std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2)
            };
            cuda.htod(d_wql, &wql)?;
            cuda.htod(d_wqh, &wqh)?;
            cuda.htod(d_wsc, as_u8(&wsc))?;
            cuda.htod(d_wd, u16_u8(&wd_bits))?;
            cuda.htod(d_xqs, as_u8(&xqs))?;
            cuda.htod_f32(d_xsc, &xsc)?;

            k.gemv_q6_k_soa(&cuda, d_wql, d_wqh, d_wsc, d_wd, d_xqs, d_xsc, d_y, out_dim as u32, in_dim as u32)?;
            cuda.synchronize()?;
            let mut y_host = vec![0f32; out_dim];
            cuda.dtoh_f32(&mut y_host, d_y)?;

            // CPU reference off the SoA streams: per 16-value sub-block,
            // acc += d*sc*xs*(dot(q6,xq) - 32*sum(xq)), kernel bit layout
            // (low4 from ql lo/hi nibble, high2 from qh byte i/4).
            let mut max_rel = 0f32;
            for r in 0..out_dim {
                let mut acc = 0f32;
                for s in 0..nsub {
                    let d = d_vals[(r * nsb + s / 16) % 4];
                    let sc = wsc[r * nsub + s] as f32;
                    let xs = xsc[s / 2];
                    let (mut dot, mut sum) = (0i32, 0i32);
                    for i in 0..16 {
                        let v = s * 16 + i;
                        let (g, rr) = (v / 8, v % 8);
                        let lbyte = wql[r * in_dim / 2 + g * 4 + (rr % 4)];
                        let hbyte = wqh[r * in_dim / 2 + g * 4 + (rr % 4)];
                        let low4 = if rr < 4 { lbyte & 0x0F } else { lbyte >> 4 };
                        let high2 = if rr < 4 { hbyte & 0x0F } else { hbyte >> 4 };
                        let q6 = ((high2 << 4) | low4) as i32;
                        let xv = xqs[v] as i32;
                        dot += q6 * xv;
                        sum += xv;
                    }
                    acc += d * sc * xs * (dot - 32 * sum) as f32;
                }
                let dl = (acc - y_host[r]).abs() / acc.abs().max(1.0);
                max_rel = max_rel.max(dl);
            }

            let iters = 100;
            let t = Instant::now();
            for _ in 0..iters {
                k.gemv_q6_k_soa(&cuda, d_wql, d_wqh, d_wsc, d_wd, d_xqs, d_xsc, d_y, out_dim as u32, in_dim as u32)?;
            }
            cuda.synchronize()?;
            let each = t.elapsed().as_secs_f64() / iters as f64;
            let sbytes = wql.len() + wqh.len() + wsc.len() + wd_bits.len() * 2;
            println!(
                "[q6k  {label}] {out_dim:>6}x{in_dim:<5}  batched {:>6.1} us ({:>5.0} GB/s of {sbytes} B) | max_rel_err {:.1e}",
                each * 1e6,
                sbytes as f64 / each / 1e9,
                max_rel,
            );
            buf.reset_to(mark);
        }
    }

    // ============================================================
    // 2e. Batched GEMM (gl_gemm_q8_0_soa) — the prefill path. Validates vs a
    //     CPU reference and times it against looping the SoA GEMV once per
    //     token (what sequential prefill does today). The GEMM streams each
    //     weight row once per 4-token tile, so it should be markedly faster.
    // ============================================================
    {
        let scale_bits: [u16; 4] = [0x3C00, 0x3800, 0x3E00, 0x3400];
        let scale_vals: [f32; 4] = [1.0, 0.5, 1.5, 0.25];
        let qb = |i: usize| -> i8 { (((i * 131 + 7) % 255) as i32 - 127) as i8 };
        let mark = buf.mark();
        let ntok = 32usize; // multiple of 4
        let (out_dim, in_dim) = (2 * hidden, dim); // gate/up shape
        let nb = in_dim / 32;

        let wqs: Vec<i8> = (0..out_dim * in_dim).map(qb).collect();
        let wsc_bits: Vec<u16> = (0..out_dim * nb).map(|i| scale_bits[i % 4]).collect();
        let xqs: Vec<i8> = (0..ntok * in_dim).map(|i| qb(i * 7 + 3)).collect();
        let xsc: Vec<f32> = (0..ntok * nb).map(|i| 0.5 + (i % 3) as f32 * 0.25).collect();

        let d_wqs = buf.alloc((out_dim * in_dim) as u64)?.dptr;
        let d_wsc = buf.alloc((out_dim * nb * 2) as u64)?.dptr;
        let d_xqs = buf.alloc((ntok * in_dim) as u64)?.dptr;
        let d_xsc = buf.alloc_f32(ntok * nb)?.dptr;
        let d_y = buf.alloc_f32(ntok * out_dim)?.dptr;
        let as_u8 = |v: &[i8]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len()) };
        let wsc_u8 =
            unsafe { std::slice::from_raw_parts(wsc_bits.as_ptr() as *const u8, wsc_bits.len() * 2) };
        cuda.htod(d_wqs, as_u8(&wqs))?;
        cuda.htod(d_wsc, wsc_u8)?;
        cuda.htod(d_xqs, as_u8(&xqs))?;
        cuda.htod_f32(d_xsc, &xsc)?;

        k.gemm_q8_0_soa(&cuda, d_wqs, d_wsc, d_xqs, d_xsc, d_y, out_dim as u32, in_dim as u32, ntok as u32)?;
        cuda.synchronize()?;
        let mut y_host = vec![0f32; ntok * out_dim];
        cuda.dtoh_f32(&mut y_host, d_y)?;

        let mut max_rel = 0f32;
        for t in 0..ntok {
            for r in 0..out_dim {
                let mut acc = 0f32;
                for b in 0..nb {
                    let mut dot = 0i32;
                    for j in 0..32 {
                        dot += wqs[r * in_dim + b * 32 + j] as i32 * xqs[t * in_dim + b * 32 + j] as i32;
                    }
                    acc += dot as f32 * (scale_vals[(r * nb + b) % 4] * xsc[t * nb + b]);
                }
                let g = y_host[t * out_dim + r];
                max_rel = max_rel.max((acc - g).abs() / acc.abs().max(1.0));
            }
        }

        let iters = 50;
        let t = Instant::now();
        for _ in 0..iters {
            k.gemm_q8_0_soa(&cuda, d_wqs, d_wsc, d_xqs, d_xsc, d_y, out_dim as u32, in_dim as u32, ntok as u32)?;
        }
        cuda.synchronize()?;
        let gemm = t.elapsed().as_secs_f64() / iters as f64;
        // baseline: N sequential GEMVs (one weight stream per token).
        let t = Instant::now();
        for _ in 0..iters {
            for tk in 0..ntok {
                let xq = d_xqs + (tk * in_dim) as u64;
                let xs = d_xsc + (tk * nb * 4) as u64;
                let yy = d_y + (tk * out_dim * 4) as u64;
                k.gemv_q8_0_soa(&cuda, d_wqs, d_wsc, xq, xs, yy, out_dim as u32, in_dim as u32)?;
            }
        }
        cuda.synchronize()?;
        let looped = t.elapsed().as_secs_f64() / iters as f64;
        println!(
            "\n[gemm gate ] {ntok}x{out_dim}x{in_dim}  batched {:.0} us | {ntok}x-gemv {:.0} us | speedup {:.2}x | max_rel_err {:.1e}",
            gemm * 1e6,
            looped * 1e6,
            looped / gemm,
            max_rel,
        );
        buf.reset_to(mark);
    }

    // ============================================================
    // 2f. Tensor-core MMA GEMM (gl_gemm_mma_q8, sm_75+) vs the dp4a GEMM on
    //     identical data — the Task B benchmark hook: kernel time reported
    //     separately per path so tensor-core utilization is a direct A/B.
    //     (Set GLCUDA_NO_MMA=1 to get the same A/B at the model level.)
    // ============================================================
    if k.has_mma() {
        let scale_bits: [u16; 4] = [0x3C00, 0x3800, 0x3E00, 0x3400];
        let scale_vals: [f32; 4] = [1.0, 0.5, 1.5, 0.25];
        let qb = |i: usize| -> i8 { (((i * 131 + 7) % 255) as i32 - 127) as i8 };
        let mark = buf.mark();
        let ntok = 32usize; // multiple of 8
        let (out_dim, in_dim) = (2 * hidden, dim);
        let nb = in_dim / 32;

        let wqs: Vec<i8> = (0..out_dim * in_dim).map(qb).collect();
        let wsc_bits: Vec<u16> = (0..out_dim * nb).map(|i| scale_bits[i % 4]).collect();
        let xqs: Vec<i8> = (0..ntok * in_dim).map(|i| qb(i * 7 + 3)).collect();
        let xsc: Vec<f32> = (0..ntok * nb).map(|i| 0.5 + (i % 3) as f32 * 0.25).collect();

        let d_wqs = buf.alloc((out_dim * in_dim) as u64)?.dptr;
        let d_wsc = buf.alloc((out_dim * nb * 2) as u64)?.dptr;
        let d_xqs = buf.alloc((ntok * in_dim) as u64)?.dptr;
        let d_xsc = buf.alloc_f32(ntok * nb)?.dptr;
        let d_y = buf.alloc_f32(ntok * out_dim)?.dptr;
        let as_u8 = |v: &[i8]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len()) };
        let wsc_u8 =
            unsafe { std::slice::from_raw_parts(wsc_bits.as_ptr() as *const u8, wsc_bits.len() * 2) };
        cuda.htod(d_wqs, as_u8(&wqs))?;
        cuda.htod(d_wsc, wsc_u8)?;
        cuda.htod(d_xqs, as_u8(&xqs))?;
        cuda.htod_f32(d_xsc, &xsc)?;

        k.gemm_mma_q8(&cuda, d_wqs, d_wsc, d_xqs, d_xsc, d_y, out_dim as u32, in_dim as u32, ntok as u32)?;
        cuda.synchronize()?;
        let mut y_host = vec![0f32; ntok * out_dim];
        cuda.dtoh_f32(&mut y_host, d_y)?;

        let mut max_rel = 0f32;
        for t in 0..ntok {
            for r in 0..out_dim {
                let mut acc = 0f32;
                for b in 0..nb {
                    let mut dot = 0i32;
                    for j in 0..32 {
                        dot += wqs[r * in_dim + b * 32 + j] as i32 * xqs[t * in_dim + b * 32 + j] as i32;
                    }
                    acc += dot as f32 * (scale_vals[(r * nb + b) % 4] * xsc[t * nb + b]);
                }
                let g = y_host[t * out_dim + r];
                max_rel = max_rel.max((acc - g).abs() / acc.abs().max(1.0));
            }
        }

        let iters = 50;
        let t = Instant::now();
        for _ in 0..iters {
            k.gemm_mma_q8(&cuda, d_wqs, d_wsc, d_xqs, d_xsc, d_y, out_dim as u32, in_dim as u32, ntok as u32)?;
        }
        cuda.synchronize()?;
        let mma = t.elapsed().as_secs_f64() / iters as f64;
        let t = Instant::now();
        for _ in 0..iters {
            k.gemm_q8_0_soa(&cuda, d_wqs, d_wsc, d_xqs, d_xsc, d_y, out_dim as u32, in_dim as u32, ntok as u32)?;
        }
        cuda.synchronize()?;
        let dp4a = t.elapsed().as_secs_f64() / iters as f64;
        let ops = 2.0 * (ntok * out_dim * in_dim) as f64;
        println!(
            "\n[mma  gate ] {ntok}x{out_dim}x{in_dim}  mma {:.0} us ({:.2} TOPS) | dp4a-gemm {:.0} us | speedup {:.2}x | max_rel_err {:.1e}",
            mma * 1e6,
            ops / mma / 1e12,
            dp4a * 1e6,
            dp4a / mma,
            max_rel,
        );
        buf.reset_to(mark);
    } else {
        println!("\n[mma] skipped (device below sm_75 or GLCUDA_NO_MMA set)");
    }

    // ============================================================
    // 2d. [gemm-reuse] Acceleratio Stellarum Phase B ceiling probe.
    //     The MMA GEMM streams each weight fragment from DRAM once and reuses
    //     it across an ntok-row m-tile (currently <= 64). Phase B proposes
    //     raising that reuse to 256 rows so weight DRAM traffic for a 512-tok
    //     chunk drops 4x. BUT the down projection's activation tile
    //     (256 x 18944) is ~5.5 MB > the T4's 4 MB L2, so its reuse may not
    //     scale. This probe measures effective WEIGHT bandwidth (weight bytes
    //     / time) at ntok = 8/16/32/64 for gate_up (in=3584, act tile fits)
    //     and down (in=18944, act tile spills). If GB/s climbs with ntok, the
    //     kernel is weight-BW-bound and more reuse pays; if it plateaus early
    //     — especially for down — that is the L2 spill capping Phase B, and
    //     the 256-row kernel is not worth 400 lines of PTX for that matmul.
    //     Measure before building.
    if k.has_mma() {
        let d7 = 3584usize;
        let h7 = 18944usize;
        let qb = |i: usize| -> i8 { (((i * 131 + 7) % 255) as i32 - 127) as i8 };
        // (label, out_dim, in_dim) at real 7B shapes.
        for (label, out_dim, in_dim) in
            [("gate_up", 2 * h7, d7), ("down   ", d7, h7)]
        {
            let mark = buf.mark();
            let nb = in_dim / 32;
            let wbytes = (out_dim * in_dim + out_dim * nb * 2) as f64; // qs + f16 scales
            let d_wqs = buf.alloc((out_dim * in_dim) as u64)?.dptr;
            let d_wsc = buf.alloc((out_dim * nb * 2) as u64)?.dptr;
            // Fill weights once (values irrelevant to timing; just not NaN).
            let wqs: Vec<i8> = (0..out_dim * in_dim).map(qb).collect();
            let as_u8 = |v: &[i8]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len()) };
            cuda.htod(d_wqs, as_u8(&wqs))?;
            cuda.htod(d_wsc, &vec![0x3C00u16; out_dim * nb].iter().flat_map(|b| b.to_le_bytes()).collect::<Vec<u8>>())?;
            // Activation + output sized for the largest ntok (64); extra rows
            // are read-safe per the kernel's round8 contract.
            let max_ntok = 64usize;
            let d_xqs = buf.alloc((max_ntok * in_dim) as u64)?.dptr;
            let d_xsc = buf.alloc_f32(max_ntok * nb)?.dptr;
            let d_y = buf.alloc_f32(max_ntok * out_dim)?.dptr;
            cuda.htod(d_xqs, as_u8(&(0..max_ntok * in_dim).map(|i| qb(i * 7 + 3)).collect::<Vec<i8>>()))?;
            cuda.htod_f32(d_xsc, &vec![0.5f32; max_ntok * nb])?;

            // The RIGHT question (v2 metric): does per-call time grow
            // proportionally with ntok? Weight bytes are fixed per call, so
            //   - compute/issue-bound  -> time ~ ntok, and time/token is FLAT
            //     while the fixed weight cost amortizes -> Phase B (more reuse)
            //     buys nothing.
            //   - weight-BW-bound       -> time barely grows (the reused weight
            //     stream dominates), so time/token FALLS steeply with ntok ->
            //     Phase B pays.
            // (The old weight-bytes/time metric divided a constant by a growing
            // time, so it fell monotonically no matter what — uninformative.)
            print!("[gemm-reuse {label}] in={in_dim:<5} wt={:.0}MB | ", wbytes / 1e6);
            let mut tpt_prev = 0.0f64;
            for (idx, &ntok) in [8u32, 16, 32, 64].iter().enumerate() {
                k.gemm_mma_q8(&cuda, d_wqs, d_wsc, d_xqs, d_xsc, d_y, out_dim as u32, in_dim as u32, ntok)?;
                cuda.synchronize()?;
                let iters = 50;
                let t = Instant::now();
                for _ in 0..iters {
                    k.gemm_mma_q8(&cuda, d_wqs, d_wsc, d_xqs, d_xsc, d_y, out_dim as u32, in_dim as u32, ntok)?;
                }
                cuda.synchronize()?;
                let each = t.elapsed().as_secs_f64() / iters as f64;
                let tpt = each * 1e6 / ntok as f64; // us per token
                // total bytes moved: weights (once) + act (int8+f32/32) + out f32.
                let abytes = (ntok as usize * in_dim + ntok as usize * (in_dim / 32) * 4) as f64;
                let obytes = (ntok as usize * out_dim * 4) as f64;
                let eff_gbs = (wbytes + abytes + obytes) / each / 1e9;
                let trend = if idx == 0 { "".to_string() } else { format!(" ({:+.0}%)", 100.0 * (tpt - tpt_prev) / tpt_prev) };
                tpt_prev = tpt;
                print!("n{ntok}: {:.1}us/tok{}  [{:.0} GB/s eff]  ", tpt, trend, eff_gbs);
            }
            println!();
            buf.reset_to(mark);
        }
        println!("[gemm-reuse] time/token FLAT across ntok => compute/issue-bound, Phase B reuse buys ~0; time/token FALLING steeply => weight-BW-bound, Phase B pays. Compare eff GB/s to the ~266 achievable.");
    }

    // ============================================================
    // 2d2. [r256-parity] r256 correctness at real 7B shapes, on hardware.
    //     The engine wire-in crashed with CUDA_ERROR_MISALIGNED_ADDRESS but
    //     the parity test only runs under `cargo test` (the notebook runs
    //     bench, not tests), so r256 has NEVER been validated on the T4. This
    //     runs it at gate_up/down shapes and ntok=256 against a CPU reference
    //     and prints max_rel_err — turning the opaque crash into a number. If
    //     it faults here, the CUDA error surfaces at the sync below.
    // ============================================================
    if k.has_mma() {
        let qb = |i: usize| -> i8 { (((i * 131 + 7) % 255) as i32 - 127) as i8 };
        // Real 7B in-dims (3584 gate_up, 18944 down); out_dim trimmed to 256
        // so the CPU reference stays cheap. ntok=256 = the r256 cap and the
        // engine's second sub-slab size. This mirrors the engine call shape.
        for (label, out_dim, in_dim, ntok) in
            [("gate_up", 256usize, 3584usize, 256usize), ("down", 256, 18944, 256)]
        {
            let mark = buf.mark();
            let nb = in_dim / 32;
            let scale_bits = 0x3C00u16; // 1.0 in f16, so scales are exactly 1.
            let wqs: Vec<i8> = (0..out_dim * in_dim).map(qb).collect();
            let xqs: Vec<i8> = (0..ntok * in_dim).map(|i| qb(i * 7 + 3)).collect();
            let as_u8 = |v: &[i8]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len()) };
            let d_wqs = buf.alloc((out_dim * in_dim) as u64)?.dptr;
            let d_wsc = buf.alloc((out_dim * nb * 2) as u64)?.dptr;
            let d_xqs = buf.alloc((ntok * in_dim) as u64)?.dptr;
            let d_xsc = buf.alloc_f32(ntok * nb)?.dptr;
            let d_y = buf.alloc_f32(ntok * out_dim)?.dptr;
            cuda.htod(d_wqs, as_u8(&wqs))?;
            cuda.htod(d_wsc, &vec![scale_bits; out_dim * nb].iter().flat_map(|b| b.to_le_bytes()).collect::<Vec<u8>>())?;
            cuda.htod(d_xqs, as_u8(&xqs))?;
            cuda.htod_f32(d_xsc, &vec![1.0f32; ntok * nb])?;

            // This launch is the suspect; the sync makes a fault visible here.
            let launched = k.gemm_mma_q8_r256(&cuda, d_wqs, d_wsc, d_xqs, d_xsc, d_y, out_dim as u32, in_dim as u32, ntok as u32);
            let synced = cuda.synchronize();
            match (launched, synced) {
                (Ok(()), Ok(())) => {
                    let mut y_host = vec![0f32; ntok * out_dim];
                    cuda.dtoh_f32(&mut y_host, d_y)?;
                    // Reference: scales all 1.0, so y[t][r] = sum_k wqs*xqs.
                    let mut max_rel = 0f32;
                    for t in 0..ntok {
                        for r in 0..out_dim {
                            let mut acc = 0i64;
                            for kk in 0..in_dim {
                                acc += wqs[r * in_dim + kk] as i64 * xqs[t * in_dim + kk] as i64;
                            }
                            let want = acc as f32;
                            let got = y_host[t * out_dim + r];
                            max_rel = max_rel.max((got - want).abs() / want.abs().max(1.0));
                        }
                    }
                    println!("[r256-parity {label}] {out_dim}x{in_dim} ntok={ntok}: OK, max_rel_err {max_rel:.2e}");
                }
                (l, s) => {
                    println!("[r256-parity {label}] FAULT: launch={l:?} sync={s:?}");
                }
            }
            buf.reset_to(mark);
        }
    }

    // ============================================================
    // 2e. [gemm-phaseb] The A/B that decides Phase B's m-tile count. The
    //     [gemm-reuse] trend says weight-BW-bound (reuse pays), but the
    //     register model warns 32 m-tiles (256 rows) drops occupancy to ~50%,
    //     which may cost achieved bandwidth on a BW-bound kernel. So process a
    //     FIXED 512-row chunk of gate_up and down THREE ways and time the
    //     whole chunk: 8 calls x 64 (the current 8-tile kernel), 4 calls x 128
    //     and 2 calls x 256 (the r256 kernel, fewer weight re-streams, lower
    //     occupancy). Winner = lowest total us for the full 512-row chunk.
    //     This pits reuse against occupancy at the real work size — the number
    //     the register model cannot predict.
    if k.has_mma() {
        let d7 = 3584usize;
        let h7 = 18944usize;
        let qb = |i: usize| -> i8 { (((i * 131 + 7) % 255) as i32 - 127) as i8 };
        let chunk = 512usize; // one layer-first ubatch
        let as_u8 = |v: &[i8]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len()) };
        for (label, out_dim, in_dim) in [("gate_up", 2 * h7, d7), ("down   ", d7, h7)] {
            let mark = buf.mark();
            let nb = in_dim / 32;
            let d_wqs = buf.alloc((out_dim * in_dim) as u64)?.dptr;
            let d_wsc = buf.alloc((out_dim * nb * 2) as u64)?.dptr;
            cuda.htod(d_wqs, as_u8(&(0..out_dim * in_dim).map(qb).collect::<Vec<i8>>()))?;
            cuda.htod(d_wsc, &vec![0x3C00u16; out_dim * nb].iter().flat_map(|b| b.to_le_bytes()).collect::<Vec<u8>>())?;
            let d_xqs = buf.alloc((chunk * in_dim) as u64)?.dptr;
            let d_xsc = buf.alloc_f32(chunk * nb)?.dptr;
            let d_y = buf.alloc_f32(chunk * out_dim)?.dptr;
            cuda.htod(d_xqs, as_u8(&(0..chunk * in_dim).map(|i| qb(i * 7 + 3)).collect::<Vec<i8>>()))?;
            cuda.htod_f32(d_xsc, &vec![0.5f32; chunk * nb])?;

            // Process the full `chunk` rows as ceil(chunk/tile) calls of `tile`
            // rows each, using the 8-tile kernel for tile<=64 and r256 above.
            let run_chunk = |tile: usize| -> Result<f64, glcore::GlError> {
                let iters = 30;
                let t = Instant::now();
                for _ in 0..iters {
                    let mut base = 0usize;
                    while base < chunk {
                        let n = (chunk - base).min(tile) as u32;
                        let xqs = d_xqs + (base * in_dim) as u64;
                        let xsc = d_xsc + (base * nb) as u64 * 4;
                        let y = d_y + (base * out_dim) as u64 * 4;
                        if tile <= 64 {
                            k.gemm_mma_q8(&cuda, d_wqs, d_wsc, xqs, xsc, y, out_dim as u32, in_dim as u32, n)?;
                        } else {
                            k.gemm_mma_q8_r256(&cuda, d_wqs, d_wsc, xqs, xsc, y, out_dim as u32, in_dim as u32, n)?;
                        }
                        base += tile;
                    }
                }
                cuda.synchronize()?;
                Ok(t.elapsed().as_secs_f64() * 1e6 / iters as f64) // us per full chunk
            };
            // Warm all three.
            for &tile in &[64usize, 128, 256] { run_chunk(tile)?; }
            let (t64, t128, t256) = (run_chunk(64)?, run_chunk(128)?, run_chunk(256)?);
            let best = if t256 <= t128 && t256 <= t64 { "256" } else if t128 <= t64 { "128" } else { "64" };
            println!(
                "[gemm-phaseb {label}] 512-row chunk: 8x64 {:.0}us | 4x128 {:.0}us ({:+.0}%) | 2x256 {:.0}us ({:+.0}%) => best tile {best}",
                t64,
                t128, 100.0 * (t128 - t64) / t64,
                t256, 100.0 * (t256 - t64) / t64,
            );
            buf.reset_to(mark);
        }
        println!("[gemm-phaseb] best tile = the m-tile count to wire into gemm_rows. 128 winning = reuse helps, occupancy holds; 64 winning = occupancy loss ate the reuse; 256 winning = max reuse wins outright.");
    }

    // ============================================================
    // 2c. The dp4a QUANTIZE TAX. Every Q8_0 matmul first quantizes its input
    //     activation (quantize_q8) — an extra kernel the dp4a path added that
    //     did not exist before. Per decode token there are ~4/layer (qkv,
    //     o_proj, gate_up, down) + 1 lm_head = 4*n_layers + 1 launches. Sum
    //     their cost at the real input sizes to see how much of the ~4.4 ms
    //     GPU token this tax is — i.e. whether killing it (fusing quantize into
    //     the producing rms_norm / silu_mul) is worth it, or whether dp4a's
    //     flat result is simply "decode GEMV is BW-bound so dp4a can't help".
    // ============================================================
    {
        let n_layers = 24usize; // Qwen2.5-0.5B
        // (input_dim, count/token) for each quantize the runner issues.
        let quants = [
            (dim, n_layers),    // attn-norm -> qkv
            (dim, n_layers),    // attn-out  -> o_proj
            (dim, n_layers),    // ffn-norm  -> gate_up
            (hidden, n_layers), // silu_mul  -> down
            (dim, 1),           // final-norm -> lm_head
        ];
        let iters = 200;
        let mut total_us = 0.0;
        for (in_dim, count) in quants {
            let mark = buf.mark();
            let x = buf.alloc_f32(in_dim)?.dptr;
            let qs = buf.alloc(in_dim as u64)?.dptr;
            let sc = buf.alloc_f32(in_dim / 32)?.dptr;
            k.quantize_q8(&cuda, x, qs, sc, in_dim as u32)?;
            cuda.synchronize()?;
            let t = Instant::now();
            for _ in 0..iters {
                k.quantize_q8(&cuda, x, qs, sc, in_dim as u32)?;
            }
            cuda.synchronize()?;
            let each = t.elapsed().as_secs_f64() / iters as f64;
            total_us += each * 1e6 * count as f64;
            buf.reset_to(mark);
        }
        println!(
            "\n[quantize tax] {} launches/token: {:.0} us/token ({:.1}% of a 4.4 ms GPU token)",
            4 * n_layers + 1,
            total_us,
            total_us / 4400.0 * 100.0,
        );
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

    // ============================================================
    // 4. [attn] Prefill attention-core PASS SPLIT (M2.3 Stage 2c evidence).
    //    The prefill profiler shows gl_attn_decode_rows_f32 = 39% of a
    //    596-token 7B prefill but cannot see inside the launch: the three
    //    passes (QK scores / softmax / V-accum) are separated by bar.sync,
    //    not by kernel boundaries. gl_attn_rows_probe is a copy of the
    //    kernel with a uniform early-exit param: stop=1 ends after the
    //    score pass, stop=2 after softmax, stop=0 runs everything.
    //    Subtraction attributes the time — at real 7B shapes (28 q heads,
    //    4 kv heads GQA, head_dim 128) on a mid-prompt 64-token chunk, so
    //    cached_len matches the profiled run's average (~298).
    // ============================================================
    {
        let n_heads = 28u32;
        let n_kv = 4usize;
        let head_dim = 128usize;
        let heads_per_kv = 7u32;
        let ntok = 64usize;
        let base = 266usize; // chunk base -> avg cached_len ~298 (mid-prompt)
        let max_seq = base + ntok;
        let head_stride = (max_seq * head_dim) as u32;

        let mark = buf.mark();
        let q = buf.alloc_f32(ntok * n_heads as usize * head_dim)?.dptr;
        let out = buf.alloc_f32(ntok * n_heads as usize * head_dim)?.dptr;
        let kc = buf.alloc_f32(n_kv * max_seq * head_dim)?.dptr;
        let vc = buf.alloc_f32(n_kv * max_seq * head_dim)?.dptr;
        let pos = buf.alloc((ntok * 4) as u64)?.dptr;

        // Deterministic, non-degenerate data so the softmax sees a real
        // distribution (timing is data-independent, but keep NaN out).
        let qh: Vec<f32> = (0..ntok * n_heads as usize * head_dim)
            .map(|i| ((i * 37 % 251) as f32 - 125.0) / 251.0)
            .collect();
        let kh: Vec<f32> = (0..n_kv * max_seq * head_dim)
            .map(|i| ((i * 53 % 241) as f32 - 120.0) / 241.0)
            .collect();
        cuda.htod_f32(q, &qh)?;
        cuda.htod_f32(kc, &kh)?;
        cuda.htod_f32(vc, &kh)?;
        let posh: Vec<u8> = (0..ntok)
            .flat_map(|t| ((base + t) as u32).to_le_bytes())
            .collect();
        cuda.htod(pos, &posh)?;

        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let probe = |stop: u32| {
            k.attn_rows_probe(
                &cuda, q, kc, vc, out, n_heads, head_dim as u32, pos, heads_per_kv,
                head_stride, scale, ntok as u32, stop,
            )
        };
        // Warm every variant (JIT + caches), plus the real kernel.
        for stop in [1u32, 2, 0] {
            probe(stop)?;
        }
        k.attn_decode_rows(
            &cuda, q, kc, vc, out, n_heads, head_dim as u32, pos, heads_per_kv,
            head_stride, scale, ntok as u32,
        )?;
        cuda.synchronize()?;

        let iters = 50;
        let time_probe = |stop: u32| -> Result<f64, glcore::GlError> {
            let t = Instant::now();
            for _ in 0..iters {
                probe(stop)?;
            }
            cuda.synchronize()?;
            Ok(t.elapsed().as_secs_f64() / iters as f64)
        };
        let t1 = time_probe(1)?; // score pass only
        let t12 = time_probe(2)?; // + softmax
        let tful = time_probe(0)?; // + V-accum (everything)
        // Cross-check that the probe copy costs the same as the real kernel.
        let t = Instant::now();
        for _ in 0..iters {
            k.attn_decode_rows(
                &cuda, q, kc, vc, out, n_heads, head_dim as u32, pos, heads_per_kv,
                head_stride, scale, ntok as u32,
            )?;
        }
        cuda.synchronize()?;
        let treal = t.elapsed().as_secs_f64() / iters as f64;

        // The naive subtraction model (pass_i = t_i - t_{i-1}) is INVALID at
        // full grid: ~4 resident blocks/SM overlap different passes, so early
        // exit raises block turnover and every in-flight block hits the
        // Pass-1 K loads at once (homogeneous contention) while the full
        // kernel's softmax/V-accum stagger neighbors' demand. So full grid
        // reports RAW times + SIGNED marginal costs only (marginal ~0 = pass
        // hidden under Pass-1 traffic).
        //
        // CAVEAT (measured, Stage 2c.1): since Pass 1 became warp-per-row,
        // the probe's full path no longer matches the real kernel at full
        // grid (T4: 1543 vs 1060 us). Identical inner loops — the mismatch
        // is the probe's THREE ret sites changing ptxas block scheduling
        // across 1792 memory-bound blocks. So the full-grid probe numbers
        // are CONTENTION INDICATORS, not absolute pass times; trust the
        // single-block ratio below for the split. The full-vs-real gap is
        // reported precisely so this stays visible, not asserted away.
        println!(
            "\n[attn] full-grid {n_heads}h x {ntok}tok, cached_len ~{}: score-only {:.0} us | +softmax {:.0} us | probe-full {:.0} us | real kernel {:.0} us",
            base + ntok / 2,
            t1 * 1e6,
            t12 * 1e6,
            tful * 1e6,
            treal * 1e6,
        );
        println!(
            "[attn]   probe-vs-real gap {:+.0} us ({:+.0}%) — probe scheduling artifact, full-grid times are contention indicators only",
            (tful - treal) * 1e6,
            100.0 * (tful - treal) / treal.max(1e-12),
        );
        println!(
            "[attn]   marginal cost at full concurrency: softmax {:+.0} us | V-accum {:+.0} us (vs score-only {:.0} us)",
            (t12 - t1) * 1e6,
            (tful - t12) * 1e6,
            t1 * 1e6,
        );

        // SINGLE BLOCK (grid 1x1): no inter-block overlap, so passes really
        // do run back-to-back and subtraction is valid — this is the true
        // intra-block pass ratio, measured as latency instead of
        // throughput. Constant launch overhead cancels in the differences.
        // pos pointer offset to the chunk middle so cached_len stays ~298.
        let pos_mid = pos + ((ntok / 2) * 4) as u64;
        for stop in [1u32, 2, 0] {
            k.attn_rows_probe(
                &cuda, q, kc, vc, out, 1, head_dim as u32, pos_mid, heads_per_kv,
                head_stride, scale, 1, stop,
            )?;
        }
        cuda.synchronize()?;
        let single_iters = 400;
        let time_single = |stop: u32| -> Result<f64, glcore::GlError> {
            let t = Instant::now();
            for _ in 0..single_iters {
                k.attn_rows_probe(
                    &cuda, q, kc, vc, out, 1, head_dim as u32, pos_mid, heads_per_kv,
                    head_stride, scale, 1, stop,
                )?;
            }
            cuda.synchronize()?;
            Ok(t.elapsed().as_secs_f64() / single_iters as f64)
        };
        let s1 = time_single(1)?;
        let s12 = time_single(2)?;
        let sful = time_single(0)?;
        let (q1, q2, q3) = (s1, s12 - s1, sful - s12);
        let spct = |x: f64| 100.0 * x / sful.max(1e-12);
        println!(
            "[attn] single-block (subtraction valid, cached_len ~{}): score(QK) {:.1} us ({:.0}%) | softmax {:+.1} us ({:.0}%) | V-accum {:+.1} us ({:.0}%) | full {:.1} us",
            base + ntok / 2,
            q1 * 1e6,
            spct(q1),
            q2 * 1e6,
            spct(q2),
            q3 * 1e6,
            spct(q3),
            sful * 1e6,
        );
        buf.reset_to(mark);
    }

    buf.free(&cuda)?;
    println!("\ndone.");
    Ok(())
}
