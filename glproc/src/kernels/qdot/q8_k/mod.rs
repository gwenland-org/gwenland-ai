//! Q8_K activation: int8 quantization in 256-element super-blocks with ONE
//! f32 scale each — the activation format Q4_K's integer-dot wants.
//!
//! # Why a second activation format exists
//!
//! The existing [`super::QuantizedActivation`] quantizes in 32-element groups,
//! one f32 scale per group. That fits the block-32 formats (Q8_0, Q5_0), but it
//! poisoned the first native Q4_K kernel: with a *different* activation scale
//! per sub-block, every sub-block's integer dot had to be scaled in the float
//! domain individually — 8 broadcast FMAs, 8 lane conversions and 8 scalar
//! offset multiplies per 256 weights. That kernel measured 1.5–2.0 GMAC/s
//! against Q8_0's 3.3 and lost 33% end-to-end.
//!
//! Q8_K's single scale per 256 elements lines up with Q4_K's super-block, so
//! the 8 sub-block scales can be applied as **integer multipliers** inside the
//! super-block and the float math collapses to two multiplies per super-block:
//!
//! ```text
//!   dot += d4·d8 · Σ_j sc_j·idot_j   −   dmin4·d8 · Σ_j m_j·bsum_j
//!                  └── all integer ──┘        └── all integer ──┘
//! ```
//!
//! `bsum_j` (the per-sub-block sum of the int8 activation) is precomputed here
//! at quantization time — once per matvec, amortized over every weight row —
//! exactly why GGML's `block_q8_K` carries `bsums` too.
//!
//! # Format facts, corrected against evidence
//!
//! The Wave-2 spec described `block_q8_K` as `{f32 d, i8 qs[256]}` = 260 bytes.
//! That omits the sums the min-term needs; GGML's real `block_q8_K` is
//! `{f32 d, i8 qs[256], i16 bsums[16]}` = **292 bytes**. This module follows
//! the GGML shape (with `i32` sums per 32-element sub-block, same information),
//! and the size assertion below pins the 292.

pub mod scalar;

/// Weights per Q8_K super-block. Matches Q4_K's super-block, on purpose.
pub const BLOCK_NUMEL: usize = 256;

/// One Q8_K super-block, GGML-compatible layout.
///
/// Kept `repr(C)` so the size assertion is meaningful; the runtime buffers in
/// [`Q8KActivation`] store the same data structure-of-arrays style instead,
/// because the kernels want contiguous `qs` streams.
#[repr(C)]
pub struct BlockQ8K {
    /// f32 scale for all 256 values (`x ≈ d * q`). f32, not f16 — the
    /// activation is quantized at runtime, so there is no storage pressure,
    /// and f16 would just add conversion error for nothing.
    pub d: f32,
    /// 256 int8 quants.
    pub qs: [i8; 256],
    /// Sum of each 16-element group of `qs` — GGML precomputes these for the
    /// affine min-term. (The runtime type below keeps per-32 sums instead:
    /// same information at Q4_K's actual sub-block granularity.)
    pub bsums: [i16; 16],
}

// The size contract, pinned. 4 + 256 + 32 = 292 — GGML's block_q8_K size.
// The spec's 260 omitted `bsums`; see the module docs.
const _: () = assert!(std::mem::size_of::<BlockQ8K>() == 292);

/// One Q4_K super-block, documented as a struct and size-pinned.
///
/// The kernels keep parsing raw GGUF bytes (they already did, and a cast would
/// buy nothing), so this exists to make the layout auditable and to hold the
/// compile-time size assertion next to the field list.
#[repr(C)]
pub struct BlockQ4K {
    /// Super-scale, f16 bits.
    pub d: u16,
    /// Super-min scale, f16 bits.
    pub dmin: u16,
    /// **8** (scale, min) pairs, **6-bit** each, packed into 12 bytes.
    ///
    /// The Wave-2 spec said "6 scales + 6 mins, 4-bit each" — that contradicts
    /// both GGML and this crate's own validated dequant path
    /// (`dequant::q4_k::scalar::scale_min`, exercised by every Q4_K model the
    /// repack loader has run correctly). 256 weights / 32 per sub-block = 8
    /// sub-blocks, each needing a scale *and* a min; 16 six-bit values = 96
    /// bits = exactly these 12 bytes.
    pub scales: [u8; 12],
    /// 256 4-bit quants, two per byte. Within each 32-byte chunk the low
    /// nibbles are one sub-block and the high nibbles the *next* — not
    /// interleaved per weight.
    pub qs: [u8; 128],
}

const _: () = assert!(std::mem::size_of::<BlockQ4K>() == 144);
const _: () = assert!(
    std::mem::size_of::<BlockQ4K>() == crate::kernels::dequant::q4_k::scalar::BLOCK_BYTES
);

/// A runtime activation vector quantized to Q8_K, structure-of-arrays.
///
/// Buffers are pre-allocated once and reused per matvec, mirroring
/// [`super::QuantizedActivation`]'s zero-alloc contract.
pub struct Q8KActivation {
    /// int8 quants, `len` valid.
    pub q: Vec<i8>,
    /// One f32 scale per 256-element super-block.
    pub d: Vec<f32>,
    /// Sum of each 32-element sub-block of `q` — the min-term input,
    /// precomputed once per quantization instead of once per weight row.
    pub bsums: Vec<i32>,
    /// Valid elements from the last `quantize` call.
    pub len: usize,
}

impl Q8KActivation {
    /// Pre-allocate for activations up to `max_len` elements.
    /// `max_len` must be a multiple of 256 — Q4_K rows always are (the GGUF
    /// quantizer falls back to block-32 formats for dimensions that are not,
    /// which is why e.g. Qwen2.5-0.5B's 896-wide tensors ship as Q5_0).
    pub fn with_capacity(max_len: usize) -> Self {
        debug_assert_eq!(max_len % BLOCK_NUMEL, 0);
        let blocks = max_len / BLOCK_NUMEL;
        Q8KActivation {
            q: vec![0; max_len],
            d: vec![0.0; blocks],
            bsums: vec![0; blocks * 8],
            len: 0,
        }
    }

    /// Quantize `x` into the pre-allocated buffers. `x.len()` must be a
    /// multiple of 256 and within capacity.
    pub fn quantize(&mut self, x: &[f32]) {
        scalar::quantize(self, x);
    }
}
