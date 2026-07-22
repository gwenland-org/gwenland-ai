//! Scalar reference dequant for GQ4A — the correctness oracle (Pridwen v5
//! §3.1 reconstruction formula, §10.2).

use glcore::format::gguf::f16_to_f32;

use super::GQ4ABlock;

/// `scale_i = super_scale * (scale_delta_i / 127.0)`;
/// `weight_f32 = scale_i * (code - 8)`, code in `[0, 15]`.
pub fn run(block: &GQ4ABlock, out: &mut [f32; 256]) {
    let super_scale = f16_to_f32(block.super_scale);
    for blk in 0..GQ4ABlock::SUB_BLOCKS {
        let scale_i = super_scale * (block.scale_delta[blk] as f32 / 127.0);
        for i in 0..GQ4ABlock::SUB_BLOCK_WEIGHTS {
            let idx = blk * GQ4ABlock::SUB_BLOCK_WEIGHTS + i;
            let byte = block.weights[idx / 2];
            let code = if idx.is_multiple_of(2) { byte & 0x0f } else { byte >> 4 };
            out[idx] = scale_i * (code as f32 - 8.0);
        }
    }
}

/// Byte-stream form: N concatenated 138-byte superblocks -> `Vec<f32>`.
pub fn run_stream(data: &[u8]) -> Vec<f32> {
    let n_blocks = data.len() / GQ4ABlock::BYTES;
    let mut out = vec![0.0f32; n_blocks * GQ4ABlock::WEIGHTS];
    for (bi, chunk) in data.chunks_exact(GQ4ABlock::BYTES).enumerate() {
        let block = GQ4ABlock::from_bytes(chunk).expect("chunks_exact guarantees exact BYTES length");
        let mut block_out = [0.0f32; 256];
        run(block, &mut block_out);
        out[bi * GQ4ABlock::WEIGHTS..(bi + 1) * GQ4ABlock::WEIGHTS].copy_from_slice(&block_out);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_block(super_scale_f32: f32, scale_delta: [i8; 8], weights: [u8; 128]) -> GQ4ABlock {
        // Encode super_scale to f16 bits via the shared bit-math (same
        // conversion glcore's f16_to_f32 inverts) — reuse via round-trip
        // through a known-good f16 constant table is overkill for a test
        // fixture, so hand-roll one exact value: 1.0 -> 0x3C00.
        let bits = if super_scale_f32 == 1.0 { 0x3C00u16 } else { panic!("test helper only supports 1.0") };
        GQ4ABlock { super_scale: bits, scale_delta, weights }
    }

    #[test]
    fn dequant_gq4a_scalar_matches_formula() {
        // super_scale = 1.0, scale_delta[0] = 127 (max) -> scale_0 = 1.0.
        // First byte 0x00 -> low nibble code 0 -> (0-8)*1.0 = -8.0;
        //                     high nibble code 0 -> -8.0 as well.
        let mut weights = [0u8; 128];
        weights[0] = 0x0F; // low nibble=15 (code 15 -> +7), high nibble=0 (code 0 -> -8)
        let mut scale_delta = [0i8; 8];
        scale_delta[0] = 127;
        let block = synthetic_block(1.0, scale_delta, weights);

        let mut out = [0.0f32; 256];
        run(&block, &mut out);

        assert!((out[0] - 7.0).abs() < 1e-6, "out[0]={}", out[0]);
        assert!((out[1] - (-8.0)).abs() < 1e-6, "out[1]={}", out[1]);
    }

    #[test]
    fn dequant_gq4a_zero_block() {
        // super_scale = 0 -> every scale_i = 0 regardless of code.
        let block = GQ4ABlock { super_scale: 0, scale_delta: [0; 8], weights: [0x88; 128] };
        let mut out = [1.0f32; 256]; // pre-fill with nonzero to prove it's overwritten
        run(&block, &mut out);
        assert!(out.iter().all(|&w| w == 0.0));
    }

    #[test]
    fn dequant_gq4a_max_code() {
        // All codes = 15 (max positive), super_scale = 1.0, all scale_delta = 127.
        let block = GQ4ABlock {
            super_scale: 0x3C00, // 1.0
            scale_delta: [127; 8],
            weights: [0xFF; 128], // both nibbles = 15
        };
        let mut out = [0.0f32; 256];
        run(&block, &mut out);
        for &w in out.iter() {
            assert!((w - 7.0).abs() < 1e-6, "w={w}");
        }
    }

    #[test]
    fn dequant_gq4a_stream_matches_per_block_calls() {
        let block = GQ4ABlock { super_scale: 0x3C00, scale_delta: [64; 8], weights: [0xAB; 128] };
        let mut raw = Vec::new();
        // Build 3 identical blocks as raw bytes.
        for _ in 0..3 {
            raw.extend_from_slice(&block.super_scale.to_le_bytes());
            for d in block.scale_delta {
                raw.push(d as u8);
            }
            raw.extend_from_slice(&block.weights);
        }
        assert_eq!(raw.len(), 3 * GQ4ABlock::BYTES);

        let stream_out = run_stream(&raw);
        let mut expected_one = [0.0f32; 256];
        run(&block, &mut expected_one);

        assert_eq!(stream_out.len(), 3 * 256);
        for i in 0..3 {
            assert_eq!(&stream_out[i * 256..(i + 1) * 256], &expected_one[..]);
        }
    }
}
