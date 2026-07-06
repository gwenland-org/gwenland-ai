use std::arch::x86_64::*;
use glcore::format::gguf::f16_to_f32;

#[target_feature(enable = "avx512f", enable = "avx512bw")]
pub unsafe fn run(data: &[u8]) -> Vec<f32> {
    let numel = (data.len() / 34) * 32;
    let mut out = vec![0.0f32; numel];
    
    for (bi, block) in data.chunks_exact(34).enumerate() {
        let d = f16_to_f32(u16::from_le_bytes(block[0..2].try_into().unwrap()));
        let d_vec = _mm512_set1_ps(d);
        let base = bi * 32;
        
        let q_ptr = block[2..].as_ptr() as *const i8;
        let out_ptr = out.as_mut_ptr().add(base);
        
        for i in 0..2 {
            // Load 16 i8s, sign extend to 16 i32s in zmm
            let q16 = _mm_loadu_si128(q_ptr.add(i * 16) as *const __m128i);
            let ints = _mm512_cvtepi8_epi32(q16);
            
            let floats = _mm512_cvtepi32_ps(ints);
            let res = _mm512_mul_ps(floats, d_vec);
            _mm512_storeu_ps(out_ptr.add(i * 16), res);
        }
    }
    out
}
