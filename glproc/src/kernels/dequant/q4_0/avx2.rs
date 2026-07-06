use std::arch::x86_64::*;
use glcore::format::gguf::f16_to_f32;

#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn run(data: &[u8]) -> Vec<f32> {
    let numel = (data.len() / 18) * 32;
    let mut out = vec![0.0f32; numel];
    
    for (bi, block) in data.chunks_exact(18).enumerate() {
        let d = f16_to_f32(u16::from_le_bytes(block[0..2].try_into().unwrap()));
        let d_vec = _mm256_set1_ps(d);
        let base = bi * 32;
        
        let q_ptr = block[2..].as_ptr();
        let out_ptr = out.as_mut_ptr().add(base);
        
        for i in 0..4 {
            let off = if i % 2 == 0 { 0 } else { 8 };
            let is_high = i >= 2;
            
            let q8 = _mm_loadl_epi64(q_ptr.add(off) as *const __m128i);
            let mut bytes = _mm256_cvtepu8_epi32(q8);
            
            if is_high {
                bytes = _mm256_srli_epi32(bytes, 4);
            }
            bytes = _mm256_and_si256(bytes, _mm256_set1_epi32(0x0f));
            bytes = _mm256_sub_epi32(bytes, _mm256_set1_epi32(8));
            
            let floats = _mm256_cvtepi32_ps(bytes);
            let res = _mm256_mul_ps(floats, d_vec);
            _mm256_storeu_ps(out_ptr.add(i * 8), res);
        }
    }
    out
}
