/// ⚠️  EXPERIMENTAL — Q4_K / Q5_K / Q6_K dequantisation paths are experimental.
///     Bit-manipulation has been verified against the GGML reference but has NOT
///     been validated against real GGUF model files end-to-end. The test helper
///     `build_q4k_block` / `build_q5k_block` only round-trips correctly for
///     sub-block scale/min values < 16. Do not use in production without
///     running `cargo test -p gwenland-core convert` and cross-checking output
///     against a known-good dequantiser (e.g. llama.cpp).
///
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
///
/// Supported quantisation formats:
///   F32, F16          — pass-through / upcast (stable)
///   Q8_0              — 8-bit symmetric, 32-element blocks (stable)
///   Q4_0              — 4-bit symmetric, 32-element blocks (stable)
///   Q4_K              — 4-bit K-quant superblock (256 elements, 8 sub-blocks) [EXPERIMENTAL]
///   Q5_K              — 5-bit K-quant superblock (256 elements, 8 sub-blocks) [EXPERIMENTAL]
///   Q6_K              — 6-bit K-quant superblock (256 elements, single scale) [EXPERIMENTAL]
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
        GgufDtype::Q4_K => match mode {
            DequantMode::Standard => dequant_q4_k_standard(&tensor.raw_data, n_elements),
            DequantMode::Euler    => dequant_q4_k_euler(&tensor.raw_data, n_elements),
        },
        GgufDtype::Q5_K => match mode {
            DequantMode::Standard => dequant_q5_k_standard(&tensor.raw_data, n_elements),
            DequantMode::Euler    => dequant_q5_k_euler(&tensor.raw_data, n_elements),
        },
        GgufDtype::Q6_K => match mode {
            DequantMode::Standard => dequant_q6_k_standard(&tensor.raw_data, n_elements),
            DequantMode::Euler    => dequant_q6_k_euler(&tensor.raw_data, n_elements),
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
// ── Q4_K — Standard ───────────────────────────────────────────────────────────

/// Q4_K standard linear dequantisation.
///
/// Superblock layout (144 bytes per 256 elements):
///   [d:  f16 (2 bytes)]  — superblock scale factor
///   [dmin: f16 (2 bytes)] — superblock min factor
///   [scales: u8 × 12 (12 bytes)] — packed 6-bit sub-block scales and mins
///   [nibbles: u8 × 128 (128 bytes)] — 4-bit packed values, 256 values total
///
/// Sub-block structure: 8 sub-blocks × 32 elements each.
/// Each sub-block has a 6-bit scale and a 6-bit min, packed into the 12-byte
/// scales region using the GGML K-quant packing scheme:
///
///   scales[0..7]  hold the low 4 bits of scale[0..7] in the low nibble
///                 and the low 4 bits of min[0..7] in the high nibble.
///   scales[8..11] hold the high 2 bits of scale[0..7] and min[0..7]
///                 packed two-per-byte.
///
/// Exact GGML packing (from ggml-quants.c get_scale_min_k4):
///   For sub-block j (0..7):
///     if j < 4:
///       scale[j] = scales[j]      & 0x3F
///       min[j]   = scales[j + 4]  & 0x3F
///     else:
///       scale[j] = (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4)
///       min[j]   = (scales[j+4] >> 4)   | ((scales[j-0] >> 6) << 4)
///
/// Dequant formula per element i in sub-block j:
///   W[i] = d * scale[j] * q[i] - dmin * min[j]
///
/// where q[i] ∈ [0, 15] is the raw 4-bit nibble (unsigned, no offset).
fn dequant_q4_k_standard(raw: &[u8], n_elements: usize) -> Result<Vec<f32>, String> {
    // Superblock: 256 elements, 8 sub-blocks of 32.
    const SUPERBLOCK_ELEMENTS: usize = 256;
    const N_SUBBLOCKS: usize = 8;
    const SUBBLOCK_ELEMENTS: usize = SUPERBLOCK_ELEMENTS / N_SUBBLOCKS; // 32

    // Layout: 2 (d) + 2 (dmin) + 12 (scales) + 128 (nibbles) = 144 bytes.
    const BLOCK_BYTES: usize = 2 + 2 + 12 + (SUPERBLOCK_ELEMENTS / 2);

    let n_blocks = (n_elements + SUPERBLOCK_ELEMENTS - 1) / SUPERBLOCK_ELEMENTS;
    if raw.len() < n_blocks * BLOCK_BYTES {
        return Err(format!(
            "Q4_K data truncated: need {} bytes, got {}",
            n_blocks * BLOCK_BYTES,
            raw.len()
        ));
    }

    let mut out = Vec::with_capacity(n_elements);

    // Reusable stack buffer for the 8 decoded (scale, min) pairs.
    let mut sub_scales = [0u8; N_SUBBLOCKS];
    let mut sub_mins   = [0u8; N_SUBBLOCKS];

    for b in 0..n_blocks {
        let base = b * BLOCK_BYTES;

        // Superblock scale and min factors (f16 → f32).
        let d    = read_f16_as_f32(&raw[base..]);
        let dmin = read_f16_as_f32(&raw[base + 2..]);

        // Decode the 12-byte packed 6-bit scales/mins region.
        // Reference: GGML get_scale_min_k4 in ggml-quants.c.
        let sc = &raw[base + 4..base + 16]; // 12 bytes
        for j in 0..N_SUBBLOCKS {
            if j < 4 {
                sub_scales[j] = sc[j]     & 0x3F;
                sub_mins[j]   = sc[j + 4] & 0x3F;
            } else {
                sub_scales[j] = (sc[j + 4] & 0x0F) | ((sc[j - 4] >> 6) << 4);
                sub_mins[j]   = (sc[j + 4] >> 4)   | ((sc[j - 0] >> 6) << 4);
            }
        }

        // Nibble data starts at byte 16 within the superblock.
        let nib_base = base + 16;

        let block_elem_count =
            SUPERBLOCK_ELEMENTS.min(n_elements - b * SUPERBLOCK_ELEMENTS);

        for j in 0..N_SUBBLOCKS {
            let scale_f = d    * sub_scales[j] as f32;
            let min_f   = dmin * sub_mins[j]   as f32;

            // Each sub-block occupies 16 nibble-bytes (32 values × 4 bits).
            // The nibble layout interleaves the first and second halves of the
            // superblock: byte k encodes element k (low nibble) and element
            // k+128 (high nibble). Within a sub-block j, the 16 bytes are at
            // offsets [j*16 .. j*16+16] in the nibble region.
            let sub_nib_start = nib_base + j * (SUBBLOCK_ELEMENTS / 2);

            let sub_elem_start = j * SUBBLOCK_ELEMENTS;
            let sub_elem_end   =
                (sub_elem_start + SUBBLOCK_ELEMENTS).min(block_elem_count);

            for k in 0..(sub_elem_end - sub_elem_start) {
                // Two values per byte: low nibble = even index, high nibble = odd.
                let byte_idx = sub_nib_start + k / 2;
                let nibble = if k % 2 == 0 {
                    raw[byte_idx] & 0x0F
                } else {
                    raw[byte_idx] >> 4
                };
                out.push(scale_f * nibble as f32 - min_f);
            }
        }
    }
    Ok(out)
}

// ── Q4_K — Euler ──────────────────────────────────────────────────────────────

/// Q4_K Euler cosine-projection dequantisation.
///
/// Decodes nibbles identically to Standard mode, then passes the raw integer
/// values through `euler_dequant_block` using the superblock `d` as δ_b.
/// The sub-block min is not applied in Euler mode — the cosine projection
/// already bounds the output range, so subtracting a per-sub-block offset
/// would break the [-0.618, 0.618] guarantee.
fn dequant_q4_k_euler(raw: &[u8], n_elements: usize) -> Result<Vec<f32>, String> {
    const SUPERBLOCK_ELEMENTS: usize = 256;
    const N_SUBBLOCKS: usize = 8;
    const SUBBLOCK_ELEMENTS: usize = SUPERBLOCK_ELEMENTS / N_SUBBLOCKS;
    const BLOCK_BYTES: usize = 2 + 2 + 12 + (SUPERBLOCK_ELEMENTS / 2);

    let n_blocks = (n_elements + SUPERBLOCK_ELEMENTS - 1) / SUPERBLOCK_ELEMENTS;
    if raw.len() < n_blocks * BLOCK_BYTES {
        return Err(format!(
            "Q4_K data truncated: need {} bytes, got {}",
            n_blocks * BLOCK_BYTES,
            raw.len()
        ));
    }

    let mut out = Vec::with_capacity(n_elements);
    let mut ivalues = Vec::with_capacity(SUPERBLOCK_ELEMENTS);

    for b in 0..n_blocks {
        let base = b * BLOCK_BYTES;
        let delta_b = read_f16_as_f32(&raw[base..]);
        let nib_base = base + 16;

        let block_elem_count =
            SUPERBLOCK_ELEMENTS.min(n_elements - b * SUPERBLOCK_ELEMENTS);

        ivalues.clear();
        for j in 0..N_SUBBLOCKS {
            let sub_nib_start = nib_base + j * (SUBBLOCK_ELEMENTS / 2);
            let sub_elem_start = j * SUBBLOCK_ELEMENTS;
            let sub_elem_end   =
                (sub_elem_start + SUBBLOCK_ELEMENTS).min(block_elem_count);

            for k in 0..(sub_elem_end - sub_elem_start) {
                let byte_idx = sub_nib_start + k / 2;
                let nibble = if k % 2 == 0 {
                    raw[byte_idx] & 0x0F
                } else {
                    raw[byte_idx] >> 4
                };
                ivalues.push(nibble as i32);
            }
        }

        let weights = euler_dequant_block(&ivalues, delta_b);
        out.extend_from_slice(&weights);
    }
    Ok(out)
}

// ── Q6_K — Standard ───────────────────────────────────────────────────────────

/// Q6_K standard linear dequantisation.
///
/// Superblock layout (210 bytes per 256 elements):
///   [ql:  u8 × 128 (128 bytes)] — low 4 bits of each 6-bit value (2 per byte)
///   [qh:  u8 × 64  (64 bytes)]  — high 2 bits of each 6-bit value (4 per byte)
///   [scales: i8 × 16 (16 bytes)] — per-sub-block signed scale (16 sub-blocks of 16)
///   [d:   f16 (2 bytes)]         — superblock scale factor
///
/// Bit reconstruction (GGML dequantize_row_q6_K):
///   For element i (0..255):
///     ql_byte = ql[i / 2]
///     qh_byte = qh[i / 4]
///
///     low4  = (ql_byte >> ((i & 1) * 4)) & 0x0F
///     high2 = (qh_byte >> ((i & 3) * 2)) & 0x03
///
///     q6_raw = low4 | (high2 << 4)          — unsigned [0, 63]
///     q      = q6_raw as i8 - 32            — signed   [-32, 31]
///
/// Sub-block structure: 16 sub-blocks × 16 elements each.
/// Each sub-block has one signed i8 scale stored in `scales[j]`.
///
/// Dequant formula:
///   W[i] = d * scales[j] * q[i]
///
/// where j = i / 16 is the sub-block index.
fn dequant_q6_k_standard(raw: &[u8], n_elements: usize) -> Result<Vec<f32>, String> {
    const SUPERBLOCK_ELEMENTS: usize = 256;
    const N_SUBBLOCKS: usize = 16;
    const SUBBLOCK_ELEMENTS: usize = SUPERBLOCK_ELEMENTS / N_SUBBLOCKS; // 16

    // Layout: 128 (ql) + 64 (qh) + 16 (scales) + 2 (d) = 210 bytes.
    const BLOCK_BYTES: usize = 128 + 64 + 16 + 2;

    let n_blocks = (n_elements + SUPERBLOCK_ELEMENTS - 1) / SUPERBLOCK_ELEMENTS;
    if raw.len() < n_blocks * BLOCK_BYTES {
        return Err(format!(
            "Q6_K data truncated: need {} bytes, got {}",
            n_blocks * BLOCK_BYTES,
            raw.len()
        ));
    }

    let mut out = Vec::with_capacity(n_elements);

    for b in 0..n_blocks {
        let base = b * BLOCK_BYTES;

        // Region offsets within the superblock.
        let ql_base     = base;           // 128 bytes: low 4 bits
        let qh_base     = base + 128;     // 64 bytes:  high 2 bits
        let scales_base = base + 192;     // 16 bytes:  i8 sub-block scales
        let d_base      = base + 208;     // 2 bytes:   f16 superblock scale

        let d = read_f16_as_f32(&raw[d_base..]);

        let block_elem_count =
            SUPERBLOCK_ELEMENTS.min(n_elements - b * SUPERBLOCK_ELEMENTS);

        for i in 0..block_elem_count {
            // ── Reconstruct 6-bit value ───────────────────────────────────────
            // ql stores two 4-bit low halves per byte (low nibble = even index).
            let ql_byte = raw[ql_base + i / 2];
            let low4    = (ql_byte >> ((i & 1) * 4)) & 0x0F;

            // qh stores four 2-bit high parts per byte.
            let qh_byte = raw[qh_base + i / 4];
            let high2   = (qh_byte >> ((i & 3) * 2)) & 0x03;

            let q6_raw = low4 | (high2 << 4);          // unsigned [0, 63]
            let q      = (q6_raw as i32) - 32;         // signed   [-32, 31]

            // ── Sub-block scale ───────────────────────────────────────────────
            let j       = i / SUBBLOCK_ELEMENTS;
            let scale_j = raw[scales_base + j] as i8 as f32;

            out.push(d * scale_j * q as f32);
        }
    }
    Ok(out)
}

// ── Q6_K — Euler ──────────────────────────────────────────────────────────────

/// Q6_K Euler cosine-projection dequantisation.
///
/// Reconstructs the signed 6-bit integers identically to Standard mode, then
/// passes them through `euler_dequant_block` using the superblock `d` as δ_b.
/// Sub-block scales are not applied — the cosine projection absorbs magnitude.
fn dequant_q6_k_euler(raw: &[u8], n_elements: usize) -> Result<Vec<f32>, String> {
    const SUPERBLOCK_ELEMENTS: usize = 256;
    const BLOCK_BYTES: usize = 128 + 64 + 16 + 2;

    let n_blocks = (n_elements + SUPERBLOCK_ELEMENTS - 1) / SUPERBLOCK_ELEMENTS;
    if raw.len() < n_blocks * BLOCK_BYTES {
        return Err(format!(
            "Q6_K data truncated: need {} bytes, got {}",
            n_blocks * BLOCK_BYTES,
            raw.len()
        ));
    }

    let mut out = Vec::with_capacity(n_elements);
    let mut ivalues = Vec::with_capacity(SUPERBLOCK_ELEMENTS);

    for b in 0..n_blocks {
        let base        = b * BLOCK_BYTES;
        let ql_base     = base;
        let qh_base     = base + 128;
        let d_base      = base + 208;

        let delta_b = read_f16_as_f32(&raw[d_base..]);

        let block_elem_count =
            SUPERBLOCK_ELEMENTS.min(n_elements - b * SUPERBLOCK_ELEMENTS);

        ivalues.clear();
        for i in 0..block_elem_count {
            let ql_byte = raw[ql_base + i / 2];
            let low4    = (ql_byte >> ((i & 1) * 4)) & 0x0F;
            let qh_byte = raw[qh_base + i / 4];
            let high2   = (qh_byte >> ((i & 3) * 2)) & 0x03;
            let q6_raw  = low4 | (high2 << 4);
            let q       = (q6_raw as i32) - 32;
            ivalues.push(q);
        }

        let weights = euler_dequant_block(&ivalues, delta_b);
        out.extend_from_slice(&weights);
    }
    Ok(out)
}

// ── Q5_K — Standard ───────────────────────────────────────────────────────────

/// Q5_K standard linear dequantisation.
///
/// Superblock layout (176 bytes per 256 elements):
///   [d:     f16 (2 bytes)]        — superblock scale factor
///   [dmin:  f16 (2 bytes)]        — superblock min factor
///   [scales: u8 × 12 (12 bytes)] — same 6-bit packed format as Q4_K
///   [qh:    u8 × 32 (32 bytes)]  — high (5th) bit for each of the 256 values
///   [ql:    u8 × 128 (128 bytes)] — low 4 bits for each of the 256 values
///
/// Sub-block structure: 8 sub-blocks × 32 elements each.
/// Scales/mins decoded identically to Q4_K (get_scale_min_k4).
///
/// Bit reconstruction for element i in sub-block j:
///   low4  = nibble from ql (same layout as Q4_K)
///   bit5  = (qh[i / 8] >> (i % 8)) & 1   — one bit per element, packed 8/byte
///   q5    = low4 | (bit5 << 4)            — unsigned [0, 31]
///
/// Dequant formula:
///   W[i] = d * scale[j] * q5[i] - dmin * min[j]
fn dequant_q5_k_standard(raw: &[u8], n_elements: usize) -> Result<Vec<f32>, String> {
    const SUPERBLOCK_ELEMENTS: usize = 256;
    const N_SUBBLOCKS: usize = 8;
    const SUBBLOCK_ELEMENTS: usize = SUPERBLOCK_ELEMENTS / N_SUBBLOCKS; // 32

    // Layout: 2 (d) + 2 (dmin) + 12 (scales) + 32 (qh) + 128 (ql) = 176 bytes.
    const BLOCK_BYTES: usize = 2 + 2 + 12 + 32 + 128;

    let n_blocks = (n_elements + SUPERBLOCK_ELEMENTS - 1) / SUPERBLOCK_ELEMENTS;
    if raw.len() < n_blocks * BLOCK_BYTES {
        return Err(format!(
            "Q5_K data truncated: need {} bytes, got {}",
            n_blocks * BLOCK_BYTES,
            raw.len()
        ));
    }

    let mut out = Vec::with_capacity(n_elements);

    // Reusable stack buffers for decoded sub-block scales and mins.
    let mut sub_scales = [0u8; N_SUBBLOCKS];
    let mut sub_mins   = [0u8; N_SUBBLOCKS];

    for b in 0..n_blocks {
        let base = b * BLOCK_BYTES;

        // Superblock scale and min (f16 → f32).
        let d    = read_f16_as_f32(&raw[base..]);
        let dmin = read_f16_as_f32(&raw[base + 2..]);

        // Decode 12-byte packed 6-bit scales/mins (identical to Q4_K).
        let sc = &raw[base + 4..base + 16];
        for j in 0..N_SUBBLOCKS {
            if j < 4 {
                sub_scales[j] = sc[j]     & 0x3F;
                sub_mins[j]   = sc[j + 4] & 0x3F;
            } else {
                sub_scales[j] = (sc[j + 4] & 0x0F) | ((sc[j - 4] >> 6) << 4);
                sub_mins[j]   = (sc[j + 4] >> 4)   | ((sc[j - 0] >> 6) << 4);
            }
        }

        // Region offsets within the superblock.
        let qh_base  = base + 16;   // 32 bytes: high bits (1 bit per element, 8/byte)
        let ql_base  = base + 48;   // 128 bytes: low 4 bits (2 per byte)

        let block_elem_count =
            SUPERBLOCK_ELEMENTS.min(n_elements - b * SUPERBLOCK_ELEMENTS);

        for j in 0..N_SUBBLOCKS {
            let scale_f = d    * sub_scales[j] as f32;
            let min_f   = dmin * sub_mins[j]   as f32;

            let sub_elem_start = j * SUBBLOCK_ELEMENTS;
            let sub_elem_end   =
                (sub_elem_start + SUBBLOCK_ELEMENTS).min(block_elem_count);

            for k in 0..(sub_elem_end - sub_elem_start) {
                // Global element index within the superblock.
                let i = sub_elem_start + k;

                // Low 4 bits: same nibble layout as Q4_K.
                let ql_byte = ql_base + i / 2;
                let low4 = if i % 2 == 0 {
                    raw[ql_byte] & 0x0F
                } else {
                    raw[ql_byte] >> 4
                };

                // High (5th) bit: 1 bit per element, 8 elements per byte.
                let qh_byte = qh_base + i / 8;
                let bit5    = (raw[qh_byte] >> (i % 8)) & 1;

                let q5 = low4 | (bit5 << 4); // unsigned [0, 31]
                out.push(scale_f * q5 as f32 - min_f);
            }
        }
    }
    Ok(out)
}

// ── Q5_K — Euler ──────────────────────────────────────────────────────────────

/// Q5_K Euler cosine-projection dequantisation.
///
/// Reconstructs the 5-bit unsigned integers identically to Standard mode, then
/// passes them through `euler_dequant_block` using the superblock `d` as δ_b.
/// Sub-block mins are not applied in Euler mode.
fn dequant_q5_k_euler(raw: &[u8], n_elements: usize) -> Result<Vec<f32>, String> {
    const SUPERBLOCK_ELEMENTS: usize = 256;
    const N_SUBBLOCKS: usize = 8;
    const SUBBLOCK_ELEMENTS: usize = SUPERBLOCK_ELEMENTS / N_SUBBLOCKS;
    const BLOCK_BYTES: usize = 2 + 2 + 12 + 32 + 128;

    let n_blocks = (n_elements + SUPERBLOCK_ELEMENTS - 1) / SUPERBLOCK_ELEMENTS;
    if raw.len() < n_blocks * BLOCK_BYTES {
        return Err(format!(
            "Q5_K data truncated: need {} bytes, got {}",
            n_blocks * BLOCK_BYTES,
            raw.len()
        ));
    }

    let mut out = Vec::with_capacity(n_elements);
    let mut ivalues = Vec::with_capacity(SUPERBLOCK_ELEMENTS);

    for b in 0..n_blocks {
        let base     = b * BLOCK_BYTES;
        let delta_b  = read_f16_as_f32(&raw[base..]);
        let qh_base  = base + 16;
        let ql_base  = base + 48;

        let block_elem_count =
            SUPERBLOCK_ELEMENTS.min(n_elements - b * SUPERBLOCK_ELEMENTS);

        ivalues.clear();
        for j in 0..N_SUBBLOCKS {
            let sub_elem_start = j * SUBBLOCK_ELEMENTS;
            let sub_elem_end   =
                (sub_elem_start + SUBBLOCK_ELEMENTS).min(block_elem_count);

            for k in 0..(sub_elem_end - sub_elem_start) {
                let i = sub_elem_start + k;

                let ql_byte = ql_base + i / 2;
                let low4 = if i % 2 == 0 {
                    raw[ql_byte] & 0x0F
                } else {
                    raw[ql_byte] >> 4
                };

                let qh_byte = qh_base + i / 8;
                let bit5    = (raw[qh_byte] >> (i % 8)) & 1;

                let q5 = low4 | (bit5 << 4);
                ivalues.push(q5 as i32);
            }
        }

        let weights = euler_dequant_block(&ivalues, delta_b);
        out.extend_from_slice(&weights);
    }
    Ok(out)
}

// ── Euler block kernel ────────────────────────────────────────────────────────

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

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Encode a u16 as two little-endian bytes (used to embed f16 bit patterns).
    fn le16(v: u16) -> [u8; 2] {
        v.to_le_bytes()
    }

    /// Encode a 16-bit float value `v` as an f16 bit pattern (nearest, no rounding).
    /// Only used for small exact values (0.0, 1.0, 2.0) where f16 is exact.
    fn f32_to_f16_bits(v: f32) -> u16 {
        // Fast path for the handful of exact values used in tests.
        if v == 0.0 { return 0x0000; }
        if v == 1.0 { return 0x3C00; }
        if v == 2.0 { return 0x4000; }
        if v == 0.5 { return 0x3800; }
        // General path: manual IEEE 754 f32→f16 conversion (round-to-zero).
        let bits = v.to_bits();
        let sign = (bits >> 31) as u16;
        let exp  = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
        let mant = (bits >> 13) & 0x3FF;
        if exp <= 0 { return sign << 15; }
        if exp >= 31 { return (sign << 15) | 0x7C00; }
        (sign << 15) | ((exp as u16) << 10) | mant as u16
    }

    // ── Q4_K tests ────────────────────────────────────────────────────────────

    /// Build a minimal single-superblock Q4_K raw buffer.
    ///
    /// Implements the exact inverse of GGML `get_scale_min_k4` so the decoder
    /// round-trips correctly for all 6-bit values [0, 63].
    ///
    /// Decoder reads (from dequant_q4_k_standard):
    ///   j < 4:  scale[j] = sc[j]     & 0x3F
    ///           min[j]   = sc[j+4]   & 0x3F
    ///   j >= 4: scale[j] = (sc[j+4] & 0x0F) | ((sc[j-4] >> 6) << 4)
    ///           min[j]   = (sc[j+4] >> 4)   | ((sc[j]   >> 6) << 4)
    ///
    /// Inverse packer:
    ///   sc[j]     (j<4) = (scale[j] & 0x3F) | ((scale[j+4] >> 4) << 6)
    ///   sc[j+4]   (j<4) = (min[j]   & 0x3F) | ((min[j+4]   >> 4) << 6)
    ///   sc[j+4+4] (j<4) = (scale[j+4] & 0x0F) | ((min[j+4] & 0x0F) << 4)
    fn build_q4k_block(
        d_val: f32,
        dmin_val: f32,
        sub_scales: &[u8; 8],
        sub_mins: &[u8; 8],
        nibbles: &[u8; 128],
    ) -> Vec<u8> {
        let mut buf = vec![0u8; 144];
        // d and dmin
        let d_bits = le16(f32_to_f16_bits(d_val));
        buf[0] = d_bits[0]; buf[1] = d_bits[1];
        let dm_bits = le16(f32_to_f16_bits(dmin_val));
        buf[2] = dm_bits[0]; buf[3] = dm_bits[1];

        // sc[0..4]: low 6 bits of scale[0..4], high 2 bits = high 2 bits of scale[4..8]
        // sc[4..8]: low 6 bits of min[0..4],   high 2 bits = high 2 bits of min[4..8]
        // sc[8..12]: low 4 bits of scale[4..8] | high 4 bits = low 4 bits of min[4..8]
        for j in 0..4usize {
            buf[4 + j]     = (sub_scales[j] & 0x3F) | ((sub_scales[j + 4] >> 4) << 6);
            buf[4 + j + 4] = (sub_mins[j]   & 0x3F) | ((sub_mins[j + 4]   >> 4) << 6);
            buf[4 + j + 8] = (sub_scales[j + 4] & 0x0F) | ((sub_mins[j + 4] & 0x0F) << 4);
        }

        // Nibble data
        buf[16..144].copy_from_slice(nibbles);
        buf
    }

    #[test]
    fn test_q4k_zero_nibbles_zero_output() {
        // All nibbles = 0, scale = 1.0, min = 0.0 → all outputs = 0.
        let sub_scales = [1u8; 8];
        let sub_mins   = [0u8; 8];
        let nibbles    = [0u8; 128];
        let raw = build_q4k_block(1.0, 0.0, &sub_scales, &sub_mins, &nibbles);
        let out = dequant_q4_k_standard(&raw, 256).unwrap();
        assert_eq!(out.len(), 256);
        for &w in &out {
            assert_eq!(w, 0.0, "expected 0.0 for zero nibbles with zero min");
        }
    }

    #[test]
    fn test_q4k_nibble_range_check() {
        // All nibbles = 0x0F (max = 15), d=1.0, dmin=0.0, all sub_scales=1, sub_mins=0.
        // Expected: W = 1.0 * 1 * 15 - 0.0 * 0 = 15.0 for every element.
        let sub_scales = [1u8; 8];
        let sub_mins   = [0u8; 8];
        let nibbles    = [0xFFu8; 128]; // each byte = low nibble 0xF, high nibble 0xF
        let raw = build_q4k_block(1.0, 0.0, &sub_scales, &sub_mins, &nibbles);
        let out = dequant_q4_k_standard(&raw, 256).unwrap();
        assert_eq!(out.len(), 256);
        for &w in &out {
            assert!(
                (w - 15.0).abs() < 1e-4,
                "expected 15.0, got {w}"
            );
        }
    }

    #[test]
    fn test_q4k_output_range_unbounded() {
        // Verify values are NOT clamped — large scale should produce large output.
        let sub_scales = [10u8; 8];
        let sub_mins   = [0u8; 8];
        let nibbles    = [0xFFu8; 128];
        let raw = build_q4k_block(2.0, 0.0, &sub_scales, &sub_mins, &nibbles);
        let out = dequant_q4_k_standard(&raw, 256).unwrap();
        // d=2.0, scale=10, q=15 → W = 2.0 * 10 * 15 = 300.0
        for &w in &out {
            assert!(
                (w - 300.0).abs() < 1e-2,
                "expected ~300.0, got {w}"
            );
        }
    }

    #[test]
    fn test_q4k_truncated_data_error() {
        let raw = vec![0u8; 10]; // far too short for one superblock
        assert!(dequant_q4_k_standard(&raw, 256).is_err());
    }

    // ── Q6_K tests ────────────────────────────────────────────────────────────

    /// Build a minimal single-superblock Q6_K raw buffer.
    ///
    /// All ql bytes set to `ql_fill`, all qh bytes set to `qh_fill`,
    /// all scales set to `scale_val`, superblock d = `d_val`.
    fn build_q6k_block(d_val: f32, ql_fill: u8, qh_fill: u8, scale_val: i8) -> Vec<u8> {
        let mut buf = vec![0u8; 210];
        // ql: bytes 0..128
        for b in buf[0..128].iter_mut() { *b = ql_fill; }
        // qh: bytes 128..192
        for b in buf[128..192].iter_mut() { *b = qh_fill; }
        // scales: bytes 192..208 (i8)
        for b in buf[192..208].iter_mut() { *b = scale_val as u8; }
        // d: bytes 208..210
        let d_bits = le16(f32_to_f16_bits(d_val));
        buf[208] = d_bits[0]; buf[209] = d_bits[1];
        buf
    }

    #[test]
    fn test_q6k_zero_ql_zero_qh_gives_minus32() {
        // ql=0x00, qh=0x00 → low4=0, high2=0 → q6_raw=0 → q=-32.
        // d=1.0, scale=1 → W = 1.0 * 1 * (-32) = -32.0.
        let raw = build_q6k_block(1.0, 0x00, 0x00, 1);
        let out = dequant_q6_k_standard(&raw, 256).unwrap();
        assert_eq!(out.len(), 256);
        for &w in &out {
            assert!(
                (w - (-32.0)).abs() < 1e-4,
                "expected -32.0, got {w}"
            );
        }
    }

    #[test]
    fn test_q6k_max_value() {
        // ql=0xFF → low4=0xF for all; qh=0xFF → high2=0x3 for all.
        // q6_raw = 0xF | (0x3 << 4) = 0x3F = 63 → q = 63 - 32 = 31.
        // d=1.0, scale=1 → W = 31.0.
        let raw = build_q6k_block(1.0, 0xFF, 0xFF, 1);
        let out = dequant_q6_k_standard(&raw, 256).unwrap();
        assert_eq!(out.len(), 256);
        for &w in &out {
            assert!(
                (w - 31.0).abs() < 1e-4,
                "expected 31.0, got {w}"
            );
        }
    }

    #[test]
    fn test_q6k_signed_range() {
        // Verify output spans [-32, 31] — the signed 6-bit range.
        // Use ql=0x00/qh=0x00 for min and ql=0xFF/qh=0xFF for max.
        let raw_min = build_q6k_block(1.0, 0x00, 0x00, 1);
        let raw_max = build_q6k_block(1.0, 0xFF, 0xFF, 1);
        let out_min = dequant_q6_k_standard(&raw_min, 256).unwrap();
        let out_max = dequant_q6_k_standard(&raw_max, 256).unwrap();
        let min_val = out_min.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = out_max.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!((min_val - (-32.0)).abs() < 1e-3, "min should be -32, got {min_val}");
        assert!((max_val - 31.0).abs() < 1e-3, "max should be 31, got {max_val}");
    }

    #[test]
    fn test_q6k_truncated_data_error() {
        let raw = vec![0u8; 50];
        assert!(dequant_q6_k_standard(&raw, 256).is_err());
    }

    // ── Q5_K tests ────────────────────────────────────────────────────────────

    /// Build a minimal single-superblock Q5_K raw buffer.
    ///
    /// All ql bytes = `ql_fill`, all qh bytes = `qh_fill`,
    /// all sub_scales = `scale_val`, all sub_mins = `min_val`,
    /// d = `d_val`, dmin = `dmin_val`.
    fn build_q5k_block(
        d_val: f32,
        dmin_val: f32,
        scale_val: u8,
        min_val: u8,
        ql_fill: u8,
        qh_fill: u8,
    ) -> Vec<u8> {
        let mut buf = vec![0u8; 176];
        // d and dmin
        let d_bits  = le16(f32_to_f16_bits(d_val));
        let dm_bits = le16(f32_to_f16_bits(dmin_val));
        buf[0] = d_bits[0];  buf[1] = d_bits[1];
        buf[2] = dm_bits[0]; buf[3] = dm_bits[1];

        // Pack scales/mins into 12 bytes — exact inverse of get_scale_min_k4.
        // Same layout as Q4_K: see build_q4k_block for the derivation.
        let sub_scales = [scale_val; 8];
        let sub_mins   = [min_val;   8];
        for j in 0..4usize {
            buf[4 + j]     = (sub_scales[j] & 0x3F) | ((sub_scales[j + 4] >> 4) << 6);
            buf[4 + j + 4] = (sub_mins[j]   & 0x3F) | ((sub_mins[j + 4]   >> 4) << 6);
            buf[4 + j + 8] = (sub_scales[j + 4] & 0x0F) | ((sub_mins[j + 4] & 0x0F) << 4);
        }

        // qh: bytes 16..48
        for b in buf[16..48].iter_mut() { *b = qh_fill; }
        // ql: bytes 48..176
        for b in buf[48..176].iter_mut() { *b = ql_fill; }
        buf
    }

    #[test]
    fn test_q5k_zero_values_zero_output() {
        // ql=0, qh=0 → q5=0; d=1.0, dmin=0.0, scale=1, min=0 → W=0.
        let raw = build_q5k_block(1.0, 0.0, 1, 0, 0x00, 0x00);
        let out = dequant_q5_k_standard(&raw, 256).unwrap();
        assert_eq!(out.len(), 256);
        for &w in &out {
            assert_eq!(w, 0.0, "expected 0.0, got {w}");
        }
    }

    #[test]
    fn test_q5k_max_value() {
        // ql=0xFF → low4=0xF; qh=0xFF → bit5=1 for all.
        // q5 = 0xF | (1 << 4) = 31.
        // d=1.0, dmin=0.0, scale=1, min=0 → W = 31.0.
        let raw = build_q5k_block(1.0, 0.0, 1, 0, 0xFF, 0xFF);
        let out = dequant_q5_k_standard(&raw, 256).unwrap();
        assert_eq!(out.len(), 256);
        for &w in &out {
            assert!(
                (w - 31.0).abs() < 1e-4,
                "expected 31.0, got {w}"
            );
        }
    }

    #[test]
    fn test_q5k_output_range_unbounded() {
        // Large scale should produce large output — no clamping.
        // d=2.0, scale=5, q5=31, min=0 → W = 2.0 * 5 * 31 = 310.0.
        let raw = build_q5k_block(2.0, 0.0, 5, 0, 0xFF, 0xFF);
        let out = dequant_q5_k_standard(&raw, 256).unwrap();
        for &w in &out {
            assert!(
                (w - 310.0).abs() < 1e-1,
                "expected ~310.0, got {w}"
            );
        }
    }

    #[test]
    fn test_q5k_min_subtraction() {
        // Verify the min term is subtracted correctly.
        // d=1.0, dmin=1.0, scale=1, min=5, ql=0, qh=0 → q5=0.
        // W = 1.0 * 1 * 0 - 1.0 * 5 = -5.0.
        let raw = build_q5k_block(1.0, 1.0, 1, 5, 0x00, 0x00);
        let out = dequant_q5_k_standard(&raw, 256).unwrap();
        for &w in &out {
            assert!(
                (w - (-5.0)).abs() < 1e-3,
                "expected -5.0, got {w}"
            );
        }
    }

    #[test]
    fn test_q5k_truncated_data_error() {
        let raw = vec![0u8; 20];
        assert!(dequant_q5_k_standard(&raw, 256).is_err());
    }

    // ── Euler mode smoke tests ─────────────────────────────────────────────────

    #[test]
    fn test_q4k_euler_output_bounded() {
        // Euler output must be in [-0.618, 0.618] for any input.
        let sub_scales = [1u8; 8];
        let sub_mins   = [0u8; 8];
        let nibbles    = [0x5Au8; 128]; // mixed nibbles
        let raw = build_q4k_block(1.0, 0.0, &sub_scales, &sub_mins, &nibbles);
        let out = dequant_q4_k_euler(&raw, 256).unwrap();
        for &w in &out {
            assert!(
                w >= -0.619 && w <= 0.619,
                "Euler output out of bounds: {w}"
            );
        }
    }

    #[test]
    fn test_q6k_euler_output_bounded() {
        let raw = build_q6k_block(1.0, 0xA5, 0x5A, 1);
        let out = dequant_q6_k_euler(&raw, 256).unwrap();
        for &w in &out {
            assert!(
                w >= -0.619 && w <= 0.619,
                "Euler output out of bounds: {w}"
            );
        }
    }

    #[test]
    fn test_q5k_euler_output_bounded() {
        let raw = build_q5k_block(1.0, 0.0, 1, 0, 0xA5, 0x5A);
        let out = dequant_q5_k_euler(&raw, 256).unwrap();
        for &w in &out {
            assert!(
                w >= -0.619 && w <= 0.619,
                "Euler output out of bounds: {w}"
            );
        }
    }
}
