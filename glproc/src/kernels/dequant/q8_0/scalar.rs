use glcore::format::gguf::f16_to_f32;

/// Weights per Q8_0 block.
pub const BLOCK_NUMEL: usize = 32;
/// Bytes per Q8_0 block (f16 scale + 32 i8 quants).
pub const BLOCK_BYTES: usize = 34;

/// Dequantize one 34-byte Q8_0 block into 32 f32 weights: `w = d * q`
/// (`output.len() >= 32`).
pub fn dequant_block(data: &[u8], output: &mut [f32]) {
    debug_assert!(data.len() >= BLOCK_BYTES);
    debug_assert!(output.len() >= 32);
    let d = f16_to_f32(u16::from_le_bytes([data[0], data[1]]));
    for (o, &b) in output.iter_mut().zip(&data[2..34]) {
        *o = (b as i8) as f32 * d;
    }
}

pub fn run(data: &[u8]) -> Vec<f32> {
    let numel = (data.len() / 34) * 32;
    let mut out = vec![0.0f32; numel];
    for (bi, block) in data.chunks_exact(34).enumerate() {
        let d = f16_to_f32(u16::from_le_bytes(block[0..2].try_into().unwrap()));
        let base = bi * 32;
        for (j, &b) in block[2..34].iter().enumerate() {
            out[base + j] = (b as i8) as f32 * d;
        }
    }
    out
}
