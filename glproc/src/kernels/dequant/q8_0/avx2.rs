use std::arch::x86_64::*;
use glcore::format::gguf::f16_to_f32;

/// Dequantize up to 8 consecutive Q8_0 blocks (`n_blocks * 34` bytes) into
/// `output[..n_blocks * 32]`. Batching amortizes the call overhead: the
/// caller dots 256 weights at a time instead of 32.
///
/// # Safety
/// Caller must ensure the CPU supports AVX2 and FMA, `n_blocks <= 8`, and
/// `data.len() >= n_blocks * 34`.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dequant_blocks(data: &[u8], n_blocks: usize, output: &mut [f32; 256]) {
    debug_assert!(n_blocks <= 8);
    for b in 0..n_blocks {
        dequant_block(&data[b * 34..], &mut output[b * 32..b * 32 + 32]);
    }
}

/// Dequantize one 34-byte Q8_0 block into 32 f32 weights: `w = d * q`
/// (`output.len() >= 32`).
///
/// # Safety
/// Caller must ensure the CPU supports AVX2 and FMA and `data.len() >= 34`.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dequant_block(data: &[u8], output: &mut [f32]) {
    debug_assert!(data.len() >= 34);
    debug_assert!(output.len() >= 32);
    let d_vec = _mm256_set1_ps(f16_to_f32(u16::from_le_bytes([data[0], data[1]])));
    let q_ptr = data.as_ptr().add(2) as *const i8;
    let out_ptr = output.as_mut_ptr();
    for i in 0..4 {
        // Load 8 i8s, sign-extend to i32, convert, scale.
        let q8 = _mm_loadl_epi64(q_ptr.add(i * 8) as *const __m128i);
        let floats = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8));
        _mm256_storeu_ps(out_ptr.add(i * 8), _mm256_mul_ps(floats, d_vec));
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn run(data: &[u8]) -> Vec<f32> {
    let numel = (data.len() / 34) * 32;
    let mut out = vec![0.0f32; numel];
    
    for (bi, block) in data.chunks_exact(34).enumerate() {
        let d = f16_to_f32(u16::from_le_bytes(block[0..2].try_into().unwrap()));
        let d_vec = _mm256_set1_ps(d);
        let base = bi * 32;
        
        let q_ptr = block[2..].as_ptr() as *const i8;
        let out_ptr = out.as_mut_ptr().add(base);
        
        for i in 0..4 {
            // Load 8 i8s, sign extend to i32s
            let q8 = _mm_loadl_epi64(q_ptr.add(i * 8) as *const __m128i);
            let ints = _mm256_cvtepi8_epi32(q8);
            
            let floats = _mm256_cvtepi32_ps(ints);
            let res = _mm256_mul_ps(floats, d_vec);
            _mm256_storeu_ps(out_ptr.add(i * 8), res);
        }
    }
    out
}
