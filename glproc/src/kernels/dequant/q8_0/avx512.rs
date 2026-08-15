use std::arch::x86_64::*;
use glcore::format::gguf::f16_to_f32;

/// # Safety
/// Requires AVX-512F and AVX-512BW. The only sanctioned caller is the
/// `SimdStrategy::Avx512` arm of `kernels::dequant_q8_0`, whose match on the
/// cached CPU probe is what proves the features are present; calling this
/// directly bypasses that proof and is UB on a CPU without them.
///
/// `data` is read as 34-byte Q8_0 blocks (2-byte f16 scale + 32 int8 weights). Any length is accepted: `chunks_exact` drops any trailing partial block, and the output is sized from that same whole-block count.
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
