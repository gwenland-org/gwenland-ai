//! Scalar reference dequant for GQ2A — the correctness oracle (Pridwen v5
//! §3.2 reconstruction formula, §10.2).

use glcore::format::gguf::f16_to_f32;

use super::GQ2ABlock;

/// `scale_i = super_scale * (1 + scale_delta_i / 7.0)`;
/// `min_i = super_min + super_scale * (min_delta_i / 7.0)`;
/// `weight_f32 = min_i + scale_i * (code / 3.0)`, code in `[0, 3]`.
pub fn run(block: &GQ2ABlock, out: &mut [f32; 256]) {
    let super_scale = f16_to_f32(block.super_scale);
    let super_min = f16_to_f32(block.super_min);
    for blk in 0..GQ2ABlock::SUB_BLOCKS {
        let scale_d = block.scale_delta_at(blk) as f32;
        let min_d = block.min_delta_at(blk) as f32;
        let scale_i = super_scale * (1.0 + scale_d / 7.0);
        let min_i = super_min + super_scale * (min_d / 7.0);
        for i in 0..GQ2ABlock::SUB_BLOCK_WEIGHTS {
            let idx = blk * GQ2ABlock::SUB_BLOCK_WEIGHTS + i;
            let code = block.weight_at(idx);
            out[idx] = min_i + scale_i * (code as f32 / 3.0);
        }
    }
}

/// Byte-stream form: N concatenated 84-byte superblocks -> `Vec<f32>`.
pub fn run_stream(data: &[u8]) -> Vec<f32> {
    let n_blocks = data.len() / GQ2ABlock::BYTES;
    let mut out = vec![0.0f32; n_blocks * GQ2ABlock::WEIGHTS];
    for (bi, chunk) in data.chunks_exact(GQ2ABlock::BYTES).enumerate() {
        let block = GQ2ABlock::from_bytes(chunk).expect("chunks_exact guarantees exact BYTES length");
        let mut block_out = [0.0f32; 256];
        run(block, &mut block_out);
        out[bi * GQ2ABlock::WEIGHTS..(bi + 1) * GQ2ABlock::WEIGHTS].copy_from_slice(&block_out);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// f16 bits for exactly 1.0, hand-rolled (same convention the GQ4A
    /// scalar tests use — avoids needing an f32_to_f16 encoder in this
    /// dequant-only test module).
    const ONE_F16: u16 = 0x3C00;

    #[test]
    fn dequant_gq2a_scalar_matches_formula() {
        // super_scale = 1.0, super_min = 0.0, scale_delta[0] = 0 -> scale_0 = 1.0,
        // min_delta[0] = 0 -> min_0 = 0.0. weight[0] code=3 (max) -> 1.0;
        // weight[1] code=0 -> 0.0.
        // byte k = w[4k] | w[4k+1]<<2 | w[4k+2]<<4 | w[4k+3]<<6, so
        // byte0 = 3 | (0<<2) = 0b0000_0011 gives w[0]=3, w[1]=0.
        let mut weights = [0u8; 64];
        weights[0] = 0b0000_0011;
        let block = GQ2ABlock {
            super_scale: ONE_F16,
            super_min: 0,
            scale_delta: [0u8; 8],
            min_delta: [0u8; 8],
            weights,
        };

        let mut out = [0.0f32; 256];
        run(&block, &mut out);

        assert!((out[0] - 1.0).abs() < 1e-6, "out[0]={}", out[0]); // code 3 -> 1.0
        assert!((out[1] - 0.0).abs() < 1e-6, "out[1]={}", out[1]); // code 0 -> 0.0
    }

    #[test]
    fn dequant_gq2a_zero_block() {
        // super_scale = 0, super_min = 0 -> every weight = 0 regardless of code.
        let block = GQ2ABlock {
            super_scale: 0,
            super_min: 0,
            scale_delta: [0u8; 8],
            min_delta: [0u8; 8],
            weights: [0xFF; 64], // all codes = 3 (max) — still must decode to 0
        };
        let mut out = [1.0f32; 256]; // pre-fill nonzero to prove it's overwritten
        run(&block, &mut out);
        assert!(out.iter().all(|&w| w == 0.0));
    }

    #[test]
    fn dequant_gq2a_asymmetric_min_offsets_every_code() {
        // super_scale = 1.0, super_min = 5.0: every weight is offset by 5.0
        // regardless of code, since scale_delta=min_delta=0 -> scale_i=1.0,
        // min_i=5.0 for every sub-block.
        let super_min_f16 = {
            // 5.0 in f16: sign=0, exp=15+2=17=0b10001, frac = (5.0/4.0 - 1.0)*1024 = 256
            // 5.0 = 1.25 * 2^2, mantissa frac = 0.25*1024 = 256 = 0x100
            0b0_10001_0100000000u16
        };
        // Don't trust the hand-derived bit pattern above without checking —
        // verify it actually decodes to 5.0 via the same f16_to_f32 the
        // kernel itself uses, before relying on it for the real assertion.
        assert!(
            (f16_to_f32(super_min_f16) - 5.0).abs() < 1e-6,
            "hand-derived f16 bit pattern for 5.0 is wrong: decodes to {}",
            f16_to_f32(super_min_f16)
        );
        let block = GQ2ABlock {
            super_scale: ONE_F16,
            super_min: super_min_f16,
            scale_delta: [0u8; 8],
            min_delta: [0u8; 8],
            weights: [0u8; 64], // all codes = 0
        };
        let mut out = [0.0f32; 256];
        run(&block, &mut out);
        for &w in out.iter() {
            assert!((w - 5.0).abs() < 1e-3, "w={w}");
        }
    }

    #[test]
    fn dequant_gq2a_stream_matches_per_block_calls() {
        let block = GQ2ABlock {
            super_scale: ONE_F16,
            super_min: 0,
            scale_delta: [0x12u8; 8],
            min_delta: [0x34u8; 8],
            weights: [0xABu8; 64],
        };
        let mut raw = Vec::new();
        for _ in 0..3 {
            raw.extend_from_slice(&block.super_scale.to_le_bytes());
            raw.extend_from_slice(&block.super_min.to_le_bytes());
            raw.extend_from_slice(&block.scale_delta);
            raw.extend_from_slice(&block.min_delta);
            raw.extend_from_slice(&block.weights);
        }
        assert_eq!(raw.len(), 3 * GQ2ABlock::BYTES);

        let stream_out = run_stream(&raw);
        let mut expected_one = [0.0f32; 256];
        run(&block, &mut expected_one);

        assert_eq!(stream_out.len(), 3 * 256);
        for i in 0..3 {
            assert_eq!(&stream_out[i * 256..(i + 1) * 256], &expected_one[..]);
        }
    }
}
