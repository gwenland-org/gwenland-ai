use std::arch::x86_64::*;
use glcore::format::gguf::f16_to_f32;

#[target_feature(enable = "avx512f", enable = "f16c")]
pub unsafe fn run(data: &[u8]) -> Vec<f32> {
    let numel = data.len() / 2;
    let mut out = vec![0.0f32; numel];
    
    let mut i = 0;
    while i + 16 <= numel {
        let f16_ptr = data[i * 2..].as_ptr() as *const __m256i;
        let floats = _mm512_cvtph_ps(_mm256_loadu_si256(f16_ptr));
        _mm512_storeu_ps(out.as_mut_ptr().add(i), floats);
        i += 16;
    }
    
    while i < numel {
        out[i] = f16_to_f32(u16::from_le_bytes(data[i * 2..i * 2 + 2].try_into().unwrap()));
        i += 1;
    }
    
    out
}
