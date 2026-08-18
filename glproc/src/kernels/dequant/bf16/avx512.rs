use std::arch::x86_64::*;

/// # Safety
/// Requires AVX-512F and AVX-512BW. The only sanctioned caller is the
/// `SimdStrategy::Avx512` arm of `kernels::dequant_bf16`, whose match on the
/// cached CPU probe is what proves the features are present; calling this
/// directly bypasses that proof and is UB on a CPU without them.
///
/// `data` is read as 2-byte bfloat16 lanes. Any length is accepted: the vector loop stops 8 lanes short of the end and a scalar tail finishes the remainder.
#[target_feature(enable = "avx512f", enable = "avx512bw")]
pub unsafe fn run(data: &[u8]) -> Vec<f32> {
    let numel = data.len() / 2;
    let mut out = vec![0.0f32; numel];
    
    let mut i = 0;
    while i + 16 <= numel {
        let bf16s = _mm256_loadu_si256(data[i * 2..].as_ptr() as *const __m256i);
        let expanded = _mm512_cvtepu16_epi32(bf16s);
        let shifted = _mm512_slli_epi32(expanded, 16);
        let floats = _mm512_castsi512_ps(shifted);
        _mm512_storeu_ps(out.as_mut_ptr().add(i), floats);
        i += 16;
    }
    
    while i < numel {
        out[i] = f32::from_bits((u16::from_le_bytes(data[i * 2..i * 2 + 2].try_into().unwrap()) as u32) << 16);
        i += 1;
    }
    
    out
}
