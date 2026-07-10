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

    buf.free(&cuda)?;
    println!("\ndone.");
    Ok(())
}
