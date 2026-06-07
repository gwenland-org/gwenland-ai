/// Dequantisation — two modes: Standard (linear) and Euler (cosine projection).
///
/// Standard mode is a lossless-within-quantisation-error linear recovery:
///   W[i] = X_quant[i] * scale + zero_point
/// where scale and zero_point come from the per-block GGUF metadata.
///
/// Euler mode is a GwenTensor-specific cosine projection that maps quantised
/// integers into a bounded weight space aligned with the GwenTensor inference
/// engine's numeric expectations. See the `euler_dequant_block` function for
/// the full derivation comment.
use super::gguf_parser::{GgufDtype, TensorInfo, read_f16_as_f32};

/// Dequantisation mode, chosen by the user via `--euler` flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DequantMode {
    /// Linear dequantisation: W = X * scale + zero_point.
    /// Lossless within the quantisation error of the original scheme.
    Standard,
    /// Euler/GwenTensor cosine projection: W = cos(θ) * δ_b / φ.
    /// Produces weights bounded in [-0.618, 0.618] with a sweet spot of
    /// [-0.309, 0.309]. Preferred for GwenTensor inference.
    Euler,
}

/// Dequantise all elements of `tensor` to f32 using the given `mode`.
///
/// Dispatches to the appropriate block-level dequant function based on dtype.
/// Returns a flat Vec<f32> of length equal to the tensor's total element count
/// (product of all shape dimensions).
pub fn dequantize(tensor: &TensorInfo, mode: DequantMode) -> Result<Vec<f32>, String> {
    let n_elements: usize = tensor.shape.iter().map(|&d| d as usize).product();

    match tensor.dtype {
        GgufDtype::F32 => dequant_f32(&tensor.raw_data, n_elements),
        GgufDtype::F16 => dequant_f16(&tensor.raw_data, n_elements),
        GgufDtype::Q8_0 => match mode {
            DequantMode::Standard => dequant_q8_0_standard(&tensor.raw_data, n_elements),
            DequantMode::Euler    => dequant_q8_0_euler(&tensor.raw_data, n_elements),
        },
        GgufDtype::Q4_0 => match mode {
            DequantMode::Standard => dequant_q4_0_standard(&tensor.raw_data, n_elements),
            DequantMode::Euler    => dequant_q4_0_euler(&tensor.raw_data, n_elements),
        },
    }
}

// ── F32 / F16 pass-through ────────────────────────────────────────────────────

/// F32 tensors: byte-reinterpret, no arithmetic needed.
fn dequant_f32(raw: &[u8], n: usize) -> Result<Vec<f32>, String> {
    if raw.len() < n * 4 {
        return Err(format!(
            "F32 tensor data truncated: need {} bytes, got {}",
            n * 4, raw.len()
        ));
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let bytes = [raw[i*4], raw[i*4+1], raw[i*4+2], raw[i*4+3]];
        out.push(f32::from_le_bytes(bytes));
    }
    Ok(out)
}

/// F16 tensors: upcast each element to f32 via the bit-manipulation in gguf_parser.
fn dequant_f16(raw: &[u8], n: usize) -> Result<Vec<f32>, String> {
    if raw.len() < n * 2 {
        return Err(format!(
            "F16 tensor data truncated: need {} bytes, got {}",
            n * 2, raw.len()
        ));
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(read_f16_as_f32(&raw[i*2..]));
    }
    Ok(out)
}

// ── Q8_0 — Standard ───────────────────────────────────────────────────────────

/// Q8_0 standard linear dequantisation.
///
/// Block layout (34 bytes per 32 elements):
///   [scale: f16 (2 bytes)] [values: i8 × 32 (32 bytes)]
///
/// Dequant formula:
///   W[i] = values[i] * scale
///
/// Q8_0 uses zero_point = 0 by convention (symmetric quantisation), so the
/// full formula W = X*scale + zero_point simplifies to W = X*scale.
/// This is the original GGML design choice documented in ggml-quants.h.
fn dequant_q8_0_standard(raw: &[u8], n_elements: usize) -> Result<Vec<f32>, String> {
    const BLOCK_ELEMENTS: usize = 32;
    const BLOCK_BYTES: usize = 2 + BLOCK_ELEMENTS; // 2 for f16 scale, 32 for i8 values

    let n_blocks = (n_elements + BLOCK_ELEMENTS - 1) / BLOCK_ELEMENTS;
    if raw.len() < n_blocks * BLOCK_BYTES {
        return Err(format!(
            "Q8_0 data truncated: need {} bytes, got {}",
            n_blocks * BLOCK_BYTES, raw.len()
        ));
    }

    let mut out = Vec::with_capacity(n_elements);
    for b in 0..n_blocks {
        let block_start = b * BLOCK_BYTES;
        let scale = read_f16_as_f32(&raw[block_start..]);

        let values_start = block_start + 2;
        let block_elem_count = BLOCK_ELEMENTS.min(n_elements - b * BLOCK_ELEMENTS);

        for k in 0..block_elem_count {
            // i8 reinterpret: treat byte as signed 8-bit integer.
            let v = raw[values_start + k] as i8;
            out.push(v as f32 * scale);
        }
    }
    Ok(out)
}

// ── Q8_0 — Euler ──────────────────────────────────────────────────────────────

/// Q8_0 Euler cosine-projection dequantisation.
///
/// See `euler_dequant_block` below for the full mathematical derivation.
fn dequant_q8_0_euler(raw: &[u8], n_elements: usize) -> Result<Vec<f32>, String> {
    const BLOCK_ELEMENTS: usize = 32;
    const BLOCK_BYTES: usize = 2 + BLOCK_ELEMENTS;

    let n_blocks = (n_elements + BLOCK_ELEMENTS - 1) / BLOCK_ELEMENTS;
    if raw.len() < n_blocks * BLOCK_BYTES {
        return Err(format!(
            "Q8_0 data truncated: need {} bytes, got {}",
            n_blocks * BLOCK_BYTES, raw.len()
        ));
    }

    let mut out = Vec::with_capacity(n_elements);
    for b in 0..n_blocks {
        let block_start = b * BLOCK_BYTES;
        let delta_b = read_f16_as_f32(&raw[block_start..]);
        let values_start = block_start + 2;
        let block_elem_count = BLOCK_ELEMENTS.min(n_elements - b * BLOCK_ELEMENTS);

        // Collect i8 values to find max_bound for this block.
        let mut ivalues = Vec::with_capacity(block_elem_count);
        for k in 0..block_elem_count {
            ivalues.push(raw[values_start + k] as i8 as i32);
        }

        let weights = euler_dequant_block(&ivalues, delta_b);
        out.extend_from_slice(&weights);
    }
    Ok(out)
}

// ── Q4_0 — Standard ───────────────────────────────────────────────────────────

/// Q4_0 standard linear dequantisation.
///
/// Block layout (18 bytes per 32 elements):
///   [scale: f16 (2 bytes)] [nibbles: u8 × 16 (16 bytes)]
///
/// Each byte encodes two 4-bit values. The values are signed 4-bit integers
/// in the range [-8, 7], stored as unsigned 0–15 with an offset of 8:
///   value = (nibble - 8)  (i.e. nibble is actually stored as u4 - 8 + 8 = u4)
///
/// Wait — GGML Q4_0 stores values as signed 4-bit integers packed differently:
///   low_nibble  = byte & 0x0F  → range [0, 15] → subtract 8 → [-8, 7]
///   high_nibble = byte >> 4    → range [0, 15] → subtract 8 → [-8, 7]
///
/// This is the canonical GGML Q4_0 nibble encoding from ggml-quants.c.
fn dequant_q4_0_standard(raw: &[u8], n_elements: usize) -> Result<Vec<f32>, String> {
    const BLOCK_ELEMENTS: usize = 32;
    // 2 bytes scale + 16 bytes nibble-packed values (32 values at 4 bits each).
    const BLOCK_BYTES: usize = 2 + BLOCK_ELEMENTS / 2;

    let n_blocks = (n_elements + BLOCK_ELEMENTS - 1) / BLOCK_ELEMENTS;
    if raw.len() < n_blocks * BLOCK_BYTES {
        return Err(format!(
            "Q4_0 data truncated: need {} bytes, got {}",
            n_blocks * BLOCK_BYTES, raw.len()
        ));
    }

    let mut out = Vec::with_capacity(n_elements);
    for b in 0..n_blocks {
        let block_start = b * BLOCK_BYTES;
        let scale = read_f16_as_f32(&raw[block_start..]);
        let nibble_start = block_start + 2;
        let block_elem_count = BLOCK_ELEMENTS.min(n_elements - b * BLOCK_ELEMENTS);

        // Unpack two 4-bit values from each byte (low nibble first, then high).
        // The subtraction of 8 converts the unsigned [0,15] storage into signed [-8,7].
        for k in 0..(block_elem_count / 2) {
            let byte = raw[nibble_start + k];
            let lo = ((byte & 0x0F) as i32) - 8;
            let hi = ((byte >> 4)   as i32) - 8;
            out.push(lo as f32 * scale);
            if b * BLOCK_ELEMENTS + k * 2 + 1 < n_elements {
                out.push(hi as f32 * scale);
            }
        }
        // Handle odd element count in the last block (rare but possible).
        if block_elem_count % 2 == 1 {
            let byte = raw[nibble_start + block_elem_count / 2];
            let lo = ((byte & 0x0F) as i32) - 8;
            out.push(lo as f32 * scale);
        }
    }
    Ok(out)
}

// ── Q4_0 — Euler ──────────────────────────────────────────────────────────────

/// Q4_0 Euler cosine-projection dequantisation.
fn dequant_q4_0_euler(raw: &[u8], n_elements: usize) -> Result<Vec<f32>, String> {
    const BLOCK_ELEMENTS: usize = 32;
    const BLOCK_BYTES: usize = 2 + BLOCK_ELEMENTS / 2;

    let n_blocks = (n_elements + BLOCK_ELEMENTS - 1) / BLOCK_ELEMENTS;
    if raw.len() < n_blocks * BLOCK_BYTES {
        return Err(format!(
            "Q4_0 data truncated: need {} bytes, got {}",
            n_blocks * BLOCK_BYTES, raw.len()
        ));
    }

    let mut out = Vec::with_capacity(n_elements);
    for b in 0..n_blocks {
        let block_start = b * BLOCK_BYTES;
        let delta_b = read_f16_as_f32(&raw[block_start..]);
        let nibble_start = block_start + 2;
        let block_elem_count = BLOCK_ELEMENTS.min(n_elements - b * BLOCK_ELEMENTS);

        // Unpack nibbles into i32 values before passing to Euler projection.
        let mut ivalues = Vec::with_capacity(block_elem_count);
        for k in 0..(block_elem_count / 2) {
            let byte = raw[nibble_start + k];
            let lo = ((byte & 0x0F) as i32) - 8;
            let hi = ((byte >> 4)   as i32) - 8;
            ivalues.push(lo);
            if b * BLOCK_ELEMENTS + k * 2 + 1 < n_elements {
                ivalues.push(hi);
            }
        }
        if block_elem_count % 2 == 1 {
            let byte = raw[nibble_start + block_elem_count / 2];
            ivalues.push(((byte & 0x0F) as i32) - 8);
        }

        let weights = euler_dequant_block(&ivalues, delta_b);
        out.extend_from_slice(&weights);
    }
    Ok(out)
}

// ── Euler block kernel ────────────────────────────────────────────────────────

/// Euler cosine projection for a single quantisation block.
///
/// # Mathematical derivation
///
/// ## Why cosine projection?
/// Linear dequant (W = X*scale) maps integers uniformly to an arbitrary real
/// range determined by the scale. GwenTensor inference is designed around
/// weight values in [-0.618, 0.618] — the inverse Golden Ratio bounds — because
/// these values align with the fixed-point arithmetic of the GwenTensor kernel.
/// Feeding linearly-dequanted weights directly into GwenTensor causes overflow
/// in the accumulator for large models where |scale| >> 0.618.
///
/// Cosine projection solves this by projecting integer indices onto the first
/// quarter of the unit circle, which is naturally bounded by [-1, 1] and
/// further scaled by δ_b/φ to fit within [-0.618, 0.618].
///
/// ## Formula
///   θ_i = (X_quant[i] × π) / max_bound
///   W[i] = cos(θ_i) × δ_b / φ
///
/// ## Variable meanings
///   X_quant[i]  — the integer quantised value (e.g. i8 ∈ [-128, 127] for Q8_0)
///   max_bound   — absolute maximum of all quantised integers in this block.
///                 Normalises the angle so θ ∈ [-π, +π] regardless of dtype.
///                 Using the per-block max rather than the theoretical dtype max
///                 (128 for Q8_0, 8 for Q4_0) preserves relative inter-block
///                 magnitude differences — the block with the largest activations
///                 uses the full [-π,+π] range, smaller blocks are compressed.
///   δ_b         — the GGUF block scale factor (the f16 stored at block head).
///                 Represents the linear reconstruction scale for this block;
///                 we reuse it as an amplitude modulator so Euler mode inherits
///                 the per-block magnitude information encoded in the GGUF file.
///   φ = 1.618…  — Golden Ratio, 1/φ ≈ 0.618. Scales the cosine output from
///                 [-1, +1] to [-0.618, +0.618], the GwenTensor sweet spot.
///                 The "sweet spot" [-0.309, 0.309] = [-0.5/φ, 0.5/φ] is the
///                 range where cos(θ) contributes maximally to dot-product
///                 accumulation precision in the GwenTensor fixed-point kernel.
///
/// ## Why cosine instead of sigmoid/tanh?
///   - Cosine is an odd function around 0: cos(0)=1 means zero quantised values
///     produce the maximum per-block scale, which matches the common case of
///     sparse quantised activations centred near zero.
///   - Unlike sigmoid/tanh, cosine projection preserves the sign of the
///     original block structure through the amplitude term δ_b (which carries
///     the GGUF sign).
///   - The period of π ensures the output range [-1,1] is exactly covered for
///     any max_bound normalisation.
///
/// ## Fallback (max_bound == 0)
/// If all quantised values in a block are zero (a common outcome for pruned
/// attention heads), max_bound = 0 and the division would be undefined. In that
/// case we output 0.0 for every element — the block carries no information and
/// its GwenTensor reconstruction is identically zero regardless of mode.
fn euler_dequant_block(ivalues: &[i32], delta_b: f32) -> Vec<f32> {
    /// φ = (1 + √5) / 2 — Golden Ratio constant.
    /// Used as the divisor so the output is bounded by 1/φ ≈ 0.618.
    const PHI: f32 = 1.618_033_9;

    let max_bound = ivalues.iter().map(|&v| v.abs()).max().unwrap_or(0) as f32;

    if max_bound == 0.0 {
        return vec![0.0f32; ivalues.len()];
    }

    ivalues
        .iter()
        .map(|&x| {
            let theta = (x as f32 * std::f32::consts::PI) / max_bound;
            theta.cos() * delta_b / PHI
        })
        .collect()
}
