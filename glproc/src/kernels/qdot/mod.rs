//! Integer-domain fused dot: quantized weights × Q8-quantized activations.
//!
//! The f32 bridge dequantizes every weight to f32 before the dot — correct,
//! but the dequant instructions dominate on 4/5-bit formats. This module
//! instead quantizes the *activation vector* to int8 once per matvec
//! (~`in_dim` ops, amortized over `out_dim` rows) and keeps the inner loop
//! in the integer domain: `_mm256_maddubs_epi16` does 32 multiply-adds per
//! instruction, and the result is scaled back to f32 once per block.
//!
//! Accuracy: activations get one 8-bit scale per 32 values (the same scheme
//! llama.cpp uses for all its k-quant matvecs); relative error per dot is
//! ~1e-3, well under the quantization noise of the weights themselves.
//!
//! Layout note: every weight format here packs its blocks so that a block's
//! weights are contiguous in the *logical* row; the activation is quantized
//! in matching 32-element groups, so block `j` of a row always pairs with
//! `q[j*32..j*32+32]`, `scales[j]`, `sums[j]`.

pub mod q5_0;
pub mod q6_k;
pub mod q8_0;

use crate::simd_strategy::SimdStrategy;

/// f16 → f32 through the F16C `vcvtph2ps` instruction. The software
/// conversion is a branchy ~15-op routine and the AVX2 kernels burn one
/// conversion per weight block (millions per token) — this is 1 instruction.
///
/// # Safety
/// CPU must support F16C. Guaranteed by `SimdStrategy::detect`, which only
/// selects a wide backend when `f16c` is present.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "f16c")]
#[inline]
pub(crate) unsafe fn f16_hw(bits: u16) -> f32 {
    use std::arch::x86_64::*;
    _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(bits as i32)))
}

/// An activation vector quantized to int8 in 32-element groups.
/// Buffers are pre-allocated once (runner workspace) and reused per matvec —
/// zero allocation in the decode loop after warm-up.
pub struct QuantizedActivation {
    /// int8 quants, `len >= in_dim`.
    pub q: Vec<i8>,
    /// One f32 scale per 32-element group: `x ≈ scale * q`.
    pub scales: Vec<f32>,
    /// Sum of the 32 int8 quants per group (for offset formats: Q5_0's -16).
    pub sums: Vec<i32>,
    /// Sum of each 16-element half-group (for Q6_K's per-16 sub-scales).
    pub sums16: Vec<i32>,
    /// Number of valid elements from the last `quantize` call.
    pub len: usize,
}

impl QuantizedActivation {
    /// Pre-allocate for activations up to `max_len` elements (`% 32 == 0`).
    pub fn with_capacity(max_len: usize) -> Self {
        let groups = max_len / 32;
        QuantizedActivation {
            q: vec![0; max_len],
            scales: vec![0.0; groups],
            sums: vec![0; groups],
            sums16: vec![0; groups * 2],
            len: 0,
        }
    }

    /// Quantize `x` (length a multiple of 32, within capacity) into the
    /// pre-allocated buffers: per 32-group, `scale = max|x| / 127` and
    /// `q_i = round(x_i / scale)`.
    pub fn quantize(&mut self, x: &[f32]) {
        debug_assert_eq!(x.len() % 32, 0);
        debug_assert!(x.len() <= self.q.len());
        self.len = x.len();
        for (g, group) in x.chunks_exact(32).enumerate() {
            let amax = group.iter().fold(0f32, |m, &v| m.max(v.abs()));
            if amax == 0.0 {
                self.scales[g] = 0.0;
                self.q[g * 32..g * 32 + 32].fill(0);
                self.sums[g] = 0;
                self.sums16[g * 2] = 0;
                self.sums16[g * 2 + 1] = 0;
                continue;
            }
            let scale = amax / 127.0;
            let inv = 127.0 / amax;
            let mut sum16 = [0i32; 2];
            for (i, &v) in group.iter().enumerate() {
                // Round half away from zero, branchless: `f32::round()` is a
                // libm call, and this loop runs for every element of every
                // activation vector in the decode loop.
                let scaled = v * inv;
                let qv = (scaled + 0.5f32.copysign(scaled)) as i32;
                self.q[g * 32 + i] = qv as i8;
                sum16[i / 16] += qv;
            }
            self.scales[g] = scale;
            self.sums[g] = sum16[0] + sum16[1];
            self.sums16[g * 2] = sum16[0];
            self.sums16[g * 2 + 1] = sum16[1];
        }
    }
}

/// True when the 256-bit EVEX VNNI dot (`vpdpbusd` on ymm) is available.
/// Detected once — this is AVX512VL+VNNI encoding-wise, but it is a 256-bit
/// datapath running at the AVX2 frequency license, so the X5 AVX-512
/// thermal ban does not apply (explicitly approved for use).
pub fn has_vnni_256() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            std::arch::is_x86_feature_detected!("avx512vnni")
                && std::arch::is_x86_feature_detected!("avx512vl")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    })
}

/// One quantized weight row · Q8 activation, integer inner loop.
/// `fmt`-specific kernels; scalar is the parity ground truth for AVX2.
pub fn row_dot_q8(
    fmt: crate::kernels::bridge::QuantFormat,
    row: &[u8],
    act: &QuantizedActivation,
    strategy: SimdStrategy,
) -> f32 {
    use crate::kernels::bridge::QuantFormat;
    // SAFETY (both arms): strategy comes from SimdStrategy::detect(), so the
    // required CPU features are present. No AVX-512 qdot kernels yet — AVX2
    // covers both wide backends.
    match fmt {
        QuantFormat::Q5_0 => match strategy {
            SimdStrategy::Avx512 | SimdStrategy::Avx2 => unsafe { q5_0::avx2::row_dot(row, act) },
            SimdStrategy::Scalar => q5_0::scalar::row_dot(row, act),
        },
        QuantFormat::Q8_0 => match strategy {
            SimdStrategy::Avx512 | SimdStrategy::Avx2 => unsafe {
                if has_vnni_256() {
                    q8_0::vnni::row_dot(row, act)
                } else {
                    q8_0::avx2::row_dot(row, act)
                }
            },
            SimdStrategy::Scalar => q8_0::scalar::row_dot(row, act),
        },
        QuantFormat::Q6K => match strategy {
            SimdStrategy::Avx512 | SimdStrategy::Avx2 => unsafe { q6_k::avx2::row_dot(row, act) },
            SimdStrategy::Scalar => q6_k::scalar::row_dot(row, act),
        },
        // Q4_K has no integer kernel yet; callers route it to the f32 bridge.
        QuantFormat::Q4K => unreachable!("Q4_K uses the f32 bridge"),
    }
}

/// True if `fmt` has an integer-domain kernel.
pub fn supports(fmt: crate::kernels::bridge::QuantFormat) -> bool {
    !matches!(fmt, crate::kernels::bridge::QuantFormat::Q4K)
}
