use std::arch::x86_64::*;

/// # Safety
/// Requires AVX2. The only sanctioned caller is the
/// `SimdStrategy::Avx2` arm of `kernels::dequant_bf16`, whose match on the
/// cached CPU probe is what proves the features are present; calling this
/// directly bypasses that proof and is UB on a CPU without them.
///
/// `data` is read as 2-byte bfloat16 lanes. Any length is accepted: the vector loop stops 8 lanes short of the end and a scalar tail finishes the remainder.
#[target_feature(enable = "avx2")]
pub unsafe fn run(data: &[u8]) -> Vec<f32> {
    let numel = data.len() / 2;
    let mut out = vec![0.0f32; numel];
    
    let mut i = 0;
    while i + 8 <= numel {
        let bf16s = _mm_loadu_si128(data[i * 2..].as_ptr() as *const __m128i);
        let expanded = _mm256_cvtepu16_epi32(bf16s);
        let shifted = _mm256_slli_epi32(expanded, 16);
        let floats = _mm256_castsi256_ps(shifted);
        _mm256_storeu_ps(out.as_mut_ptr().add(i), floats);
        i += 8;
    }
    
    while i < numel {
        out[i] = f32::from_bits((u16::from_le_bytes(data[i * 2..i * 2 + 2].try_into().unwrap()) as u32) << 16);
        i += 1;
    }
    
    out
}
