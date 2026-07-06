//! Bridge-ing: dequant → L1 buffer → matmul.
//!
//! This module orchestrates only. It does NOT implement dequant or matmul —
//! those live in `kernels/dequant/` and `kernels/matmul/`, which never
//! import each other. The pipeline per quantized block:
//!
//! ```text
//! dequant/<fmt>/{scalar,avx2,avx512} ──► [f32; N] stack buffer ──► matmul dot
//! ```
//!
//! The buffer lives on the stack (≤ 1 KB), so it stays hot in L1 cache
//! between the dequant and dot stages — weights never round-trip through
//! RAM as f32. That keeps the decode working set at the *quantized* size,
//! which is what matters on a memory-bandwidth-bound CPU.

use crate::kernels::dequant::{q4_k, q5_0, q6_k, q8_0};
use crate::kernels::matmul;
use crate::simd_strategy::SimdStrategy;

/// Quantized weight format the bridge can stream through the L1 pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)] // canonical GGML names
pub enum QuantFormat {
    /// 4-bit K-quant, 256 weights / 144 bytes.
    Q4K,
    /// 5-bit, 32 weights / 22 bytes.
    Q5_0,
    /// 6-bit K-quant, 256 weights / 210 bytes.
    Q6K,
    /// 8-bit, 32 weights / 34 bytes.
    Q8_0,
}

impl QuantFormat {
    /// Weights per quantization block.
    pub fn block_numel(self) -> usize {
        match self {
            QuantFormat::Q4K => q4_k::scalar::BLOCK_NUMEL,
            QuantFormat::Q5_0 => q5_0::scalar::BLOCK_NUMEL,
            QuantFormat::Q6K => q6_k::scalar::BLOCK_NUMEL,
            QuantFormat::Q8_0 => q8_0::scalar::BLOCK_NUMEL,
        }
    }

    /// Bytes per quantization block.
    pub fn block_bytes(self) -> usize {
        match self {
            QuantFormat::Q4K => q4_k::scalar::BLOCK_BYTES,
            QuantFormat::Q5_0 => q5_0::scalar::BLOCK_BYTES,
            QuantFormat::Q6K => q6_k::scalar::BLOCK_BYTES,
            QuantFormat::Q8_0 => q8_0::scalar::BLOCK_BYTES,
        }
    }
}

/// Dot product of `input` with the f32 slice sitting in the bridge buffer.
#[inline(always)]
fn buffer_dot(buffer: &[f32], input: &[f32], strategy: SimdStrategy) -> f32 {
    // SAFETY: `strategy` comes from SimdStrategy::detect(), so the required
    // CPU features are present.
    match strategy {
        SimdStrategy::Avx512 => unsafe { matmul::avx512::dot_f32(buffer, input) },
        SimdStrategy::Avx2 => unsafe { matmul::avx2::dot_f32(buffer, input) },
        SimdStrategy::Scalar => matmul::scalar::dot_f32(buffer, input),
    }
}

/// One Q4_K weight row (`n_blocks * 144` bytes) · `input`.
fn row_dot_q4k(row: &[u8], input: &[f32], strategy: SimdStrategy) -> f32 {
    let mut acc = 0f32;
    // Stack buffer: 1 KB, fits in L1 cache. Reused every block.
    let mut buffer = [0f32; 256];
    for (block, xs) in row.chunks_exact(144).zip(input.chunks_exact(256)) {
        // SAFETY: strategy implies the CPU features; block is 144 bytes.
        match strategy {
            SimdStrategy::Avx512 => unsafe { q4_k::avx512::dequant_block(block, &mut buffer) },
            SimdStrategy::Avx2 => unsafe { q4_k::avx2::dequant_block(block, &mut buffer) },
            SimdStrategy::Scalar => q4_k::scalar::dequant_block(block, &mut buffer),
        }
        acc += buffer_dot(&buffer, xs, strategy);
    }
    acc
}

/// One Q5_0 weight row (`n_blocks * 22` bytes) · `input`. Blocks are batched
/// 8 at a time into the 256-weight L1 buffer so each dot covers 256 weights —
/// per-block call overhead would otherwise dominate these 32-weight blocks.
fn row_dot_q5_0(row: &[u8], input: &[f32], strategy: SimdStrategy) -> f32 {
    let n_blocks = row.len() / 22;
    let mut acc = 0f32;
    let mut buffer = [0f32; 256];
    let mut b = 0;
    while b < n_blocks {
        let n = (n_blocks - b).min(8);
        // SAFETY: strategy implies the CPU features; n <= 8 and the row has
        // n more blocks. No AVX-512 Q5_0 kernel yet — AVX2 covers both.
        match strategy {
            SimdStrategy::Avx512 | SimdStrategy::Avx2 => unsafe {
                q5_0::avx2::dequant_blocks(&row[b * 22..], n, &mut buffer)
            },
            SimdStrategy::Scalar => {
                for i in 0..n {
                    q5_0::scalar::dequant_block(
                        &row[(b + i) * 22..],
                        &mut buffer[i * 32..i * 32 + 32],
                    );
                }
            }
        }
        acc += buffer_dot(&buffer[..n * 32], &input[b * 32..(b + n) * 32], strategy);
        b += n;
    }
    acc
}

/// One Q6_K weight row (`n_blocks * 210` bytes) · `input`.
fn row_dot_q6k(row: &[u8], input: &[f32], strategy: SimdStrategy) -> f32 {
    let mut acc = 0f32;
    let mut buffer = [0f32; 256];
    for (block, xs) in row.chunks_exact(210).zip(input.chunks_exact(256)) {
        // SAFETY: strategy implies the CPU features; block is 210 bytes.
        match strategy {
            SimdStrategy::Avx512 | SimdStrategy::Avx2 => unsafe {
                q6_k::avx2::dequant_block(block, &mut buffer)
            },
            SimdStrategy::Scalar => q6_k::scalar::dequant_block(block, &mut buffer),
        }
        acc += buffer_dot(&buffer, xs, strategy);
    }
    acc
}

/// One Q8_0 weight row (`n_blocks * 34` bytes) · `input`. Batched like Q5_0.
fn row_dot_q8_0(row: &[u8], input: &[f32], strategy: SimdStrategy) -> f32 {
    let n_blocks = row.len() / 34;
    let mut acc = 0f32;
    let mut buffer = [0f32; 256];
    let mut b = 0;
    while b < n_blocks {
        let n = (n_blocks - b).min(8);
        // SAFETY: strategy implies the CPU features; n <= 8 and the row has
        // n more blocks.
        match strategy {
            SimdStrategy::Avx512 | SimdStrategy::Avx2 => unsafe {
                q8_0::avx2::dequant_blocks(&row[b * 34..], n, &mut buffer)
            },
            SimdStrategy::Scalar => {
                for i in 0..n {
                    q8_0::scalar::dequant_block(
                        &row[(b + i) * 34..],
                        &mut buffer[i * 32..i * 32 + 32],
                    );
                }
            }
        }
        acc += buffer_dot(&buffer[..n * 32], &input[b * 32..(b + n) * 32], strategy);
        b += n;
    }
    acc
}

/// Dot product of one quantized weight row against `input`, for any
/// supported format. Orchestrates only; zero heap allocation.
pub fn bridge_row_dot(
    fmt: QuantFormat,
    row: &[u8],
    input: &[f32],
    strategy: SimdStrategy,
) -> f32 {
    match fmt {
        QuantFormat::Q4K => row_dot_q4k(row, input, strategy),
        QuantFormat::Q5_0 => row_dot_q5_0(row, input, strategy),
        QuantFormat::Q6K => row_dot_q6k(row, input, strategy),
        QuantFormat::Q8_0 => row_dot_q8_0(row, input, strategy),
    }
}

/// Dot product of one Q4_K-quantized weight row against `input`.
/// (Kept as the historical M1.5 entry point; see [`bridge_row_dot`].)
pub fn bridge_matmul_q4k(
    weight_blocks: &[u8],
    input: &[f32],
    n_blocks: usize,
    strategy: SimdStrategy,
) -> f32 {
    row_dot_q4k(
        &weight_blocks[..n_blocks * 144],
        &input[..n_blocks * 256],
        strategy,
    )
}

/// Matrix-vector product over a quantized weight matrix `[out_dim, in_dim]`
/// (GGUF row-major, quantization blocks run along `in_dim`).
///
/// `in_dim` must be a multiple of the format's block size. `y` must be
/// pre-allocated with `len == out_dim`.
pub fn bridge_matvec_quant(
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
    for (o, out) in y.iter_mut().enumerate() {
        let row = &weights[o * row_bytes..(o + 1) * row_bytes];
        *out = bridge_row_dot(fmt, row, x, strategy);
    }
}
