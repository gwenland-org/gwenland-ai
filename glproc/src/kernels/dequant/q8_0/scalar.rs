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

/// Quantize f32 values (`len % 32 == 0`) into Q8_0 blocks: per 32-group,
/// `d = max|v| / 127`, `q = round(v / d)`. Load-time repack helper — other
/// formats whose unpack is compute-bound convert to Q8_0 through this.
pub fn quantize(values: &[f32]) -> Vec<u8> {
    debug_assert_eq!(values.len() % BLOCK_NUMEL, 0);
    let mut out = Vec::with_capacity(values.len() / BLOCK_NUMEL * BLOCK_BYTES);
    for group in values.chunks_exact(BLOCK_NUMEL) {
        let amax = group.iter().fold(0f32, |m, &v| m.max(v.abs()));
        // Round-trip the scale through f16 so `q` is computed against the
        // exact value the dequantizer will read back.
        let d_bits = f32_to_f16_bits(amax / 127.0);
        let d = f16_to_f32(d_bits);
        out.extend_from_slice(&d_bits.to_le_bytes());
        let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
        for &v in group {
            let scaled = v * inv;
            out.push(((scaled + 0.5f32.copysign(scaled)) as i32 as i8) as u8);
        }
    }
    out
}

/// f32 → f16 bit pattern, truncating the mantissa. Fine for fresh quantizer
/// scales — sub-ulp rounding is far below the 8-bit quantization noise.
pub(crate) fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mantissa = ((bits >> 13) & 0x3FF) as u16;
    if exp <= 0 {
        sign // underflow → signed zero
    } else if exp >= 31 {
        sign | 0x7C00 // overflow → infinity
    } else {
        sign | ((exp as u16) << 10) | mantissa
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
