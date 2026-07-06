use glcore::format::gguf::f16_to_f32;

pub fn run(data: &[u8]) -> Vec<f32> {
    let numel = (data.len() / 18) * 32;
    let mut out = vec![0.0f32; numel];
    for (bi, block) in data.chunks_exact(18).enumerate() {
        let d = f16_to_f32(u16::from_le_bytes(block[0..2].try_into().unwrap()));
        let qs = &block[2..18];
        let base = bi * 32;
        for (j, &b) in qs.iter().enumerate() {
            let lo = (b & 0x0f) as i32 - 8;
            let hi = (b >> 4) as i32 - 8;
            out[base + j] = lo as f32 * d;
            out[base + j + 16] = hi as f32 * d;
        }
    }
    out
}
