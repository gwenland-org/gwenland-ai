use std::arch::x86_64::*;

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
