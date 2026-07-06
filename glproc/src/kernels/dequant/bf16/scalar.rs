

pub fn run(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|b| f32::from_bits((u16::from_le_bytes(b[0..2].try_into().unwrap()) as u32) << 16))
        .collect()
}
