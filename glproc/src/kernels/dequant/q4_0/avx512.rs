use std::arch::x86_64::*;
use glcore::format::gguf::f16_to_f32;

#[target_feature(enable = "avx512f", enable = "avx512bw")]
pub unsafe fn run(data: &[u8]) -> Vec<f32> {
    let numel = (data.len() / 18) * 32;
    let mut out = vec![0.0f32; numel];
    
    for (bi, block) in data.chunks_exact(18).enumerate() {
        let d = f16_to_f32(u16::from_le_bytes(block[0..2].try_into().unwrap()));
        let d_vec = _mm512_set1_ps(d);
        let base = bi * 32;
        
        let q_ptr = block[2..].as_ptr();
        let out_ptr = out.as_mut_ptr().add(base);
        
        for i in 0..2 {
            let is_high = i == 1;
            
            // Load 16 bytes, expand to 16 i32s in zmm
            let q16 = _mm_loadu_si128(q_ptr as *const __m128i);
            let mut bytes = _mm512_cvtepu8_epi32(q16);
            
            if is_high {
                bytes = _mm512_srli_epi32(bytes, 4);
            }
            bytes = _mm512_and_epi32(bytes, _mm512_set1_epi32(0x0f));
            bytes = _mm512_sub_epi32(bytes, _mm512_set1_epi32(8));
            
            let floats = _mm512_cvtepi32_ps(bytes);
            let res = _mm512_mul_ps(floats, d_vec);
            _mm512_storeu_ps(out_ptr.add(i * 16), res);
        }
    }
    out
}
