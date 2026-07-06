use glcore::format::gguf::f16_to_f32;

pub fn run(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|b| f16_to_f32(u16::from_le_bytes(b[0..2].try_into().unwrap())))
        .collect()
}
