//! StableHLO op emitters (ARTX02 §3).
//!
//! Pure text formatting. Every function takes [`SsaName`] operands plus the
//! [`Shape`]s they were already known to have, and returns the [`SsaName`] of
//! the result. **Nothing here infers a shape** — that is
//! [`crate::graph::FuncBuilder`]'s job, and keeping the split means a bad
//! shape is caught by one piece of code rather than by nine.
//!
//! # Syntax form
//!
//! ARTX02's opening finding is that MLIR accepts two forms and gljax emits the
//! **generic** one:
//!
//! ```text
//! generic:  %v2 = "stablehlo.add"(%v0, %v1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
//! pretty:   %v2 = stablehlo.add %v0, %v1 : tensor<f32>
//! ```
//!
//! The generic form is what the spec's own examples use, states every type
//! explicitly, and maps 1:1 to the op definition.
//!
//! # Not here yet
//!
//! `gather`, `scatter`, `while`, `all_reduce`, `custom_call`. Wave A3 adds
//! `gather` (embedding lookup) and Wave A5 `dynamic_update_slice` (KV cache).
//! Emitting ops before something needs them would mean shipping untested MLIR.

use crate::stablehlo::emitter::{MlirEmitter, SsaName};
use crate::stablehlo::types::{DType, Shape};
use glcore::error::GlError;

// ---------------------------------------------------------------------------
// Elementwise
// ---------------------------------------------------------------------------

macro_rules! emit_binary_op {
    ($fn_name:ident, $mnemonic:literal, $doc:literal) => {
        #[doc = $doc]
        ///
        /// Operand and result types are identical — StableHLO elementwise ops
        /// do not broadcast. Rank/shape reconciliation is the caller's job.
        pub fn $fn_name(e: &mut MlirEmitter, lhs: SsaName, rhs: SsaName, ty: &Shape) -> SsaName {
            let out = e.fresh();
            let t = ty.mlir_type();
            e.line(format!(
                r#"{out} = "stablehlo.{m}"({lhs}, {rhs}) : ({t}, {t}) -> {t}"#,
                m = $mnemonic
            ));
            out
        }
    };
}

emit_binary_op!(emit_add, "add", "Emits `stablehlo.add`.");
emit_binary_op!(emit_subtract, "subtract", "Emits `stablehlo.subtract`.");
emit_binary_op!(emit_multiply, "multiply", "Emits `stablehlo.multiply`.");
emit_binary_op!(emit_divide, "divide", "Emits `stablehlo.divide`.");
emit_binary_op!(emit_maximum, "maximum", "Emits `stablehlo.maximum`.");
emit_binary_op!(emit_minimum, "minimum", "Emits `stablehlo.minimum`.");

macro_rules! emit_unary_op {
    ($fn_name:ident, $mnemonic:literal, $doc:literal) => {
        #[doc = $doc]
        pub fn $fn_name(e: &mut MlirEmitter, operand: SsaName, ty: &Shape) -> SsaName {
            let out = e.fresh();
            let t = ty.mlir_type();
            e.line(format!(
                r#"{out} = "stablehlo.{m}"({operand}) : ({t}) -> {t}"#,
                m = $mnemonic
            ));
            out
        }
    };
}

emit_unary_op!(emit_negate, "negate", "Emits `stablehlo.negate`.");
emit_unary_op!(emit_rsqrt, "rsqrt", "Emits `stablehlo.rsqrt` — 1/sqrt(x).");
emit_unary_op!(emit_sqrt, "sqrt", "Emits `stablehlo.sqrt`.");
emit_unary_op!(
    emit_logistic,
    "logistic",
    "Emits `stablehlo.logistic` — the sigmoid. ARTX01 §7.5: this is the op XLA \
     knows to fuse with a following multiply, so SiLU must be built from it \
     rather than approximated with `tanh` or a piecewise form."
);
emit_unary_op!(
    emit_exponential,
    "exponential",
    "Emits `stablehlo.exponential`. ⚠️ The mnemonic is `exponential`, not `exp`."
);
emit_unary_op!(emit_log, "log", "Emits `stablehlo.log`.");
emit_unary_op!(emit_tanh, "tanh", "Emits `stablehlo.tanh`.");
emit_unary_op!(emit_abs, "abs", "Emits `stablehlo.abs`.");

// ---------------------------------------------------------------------------
// Convert + constant
// ---------------------------------------------------------------------------

/// Emits `stablehlo.convert` — the only way to change dtype.
///
/// StableHLO has no implicit casting (ARTX01 §3.3): every dtype boundary is an
/// explicit op. `in_shape` and `out_shape` must have the same dims.
pub fn emit_convert(
    e: &mut MlirEmitter,
    operand: SsaName,
    in_shape: &Shape,
    out_shape: &Shape,
) -> SsaName {
    let out = e.fresh();
    e.line(format!(
        r#"{out} = "stablehlo.convert"({operand}) : ({}) -> {}"#,
        in_shape.mlir_type(),
        out_shape.mlir_type()
    ));
    out
}

/// Emits a splat `stablehlo.constant` — every element set to `value`.
///
/// Refuses non-finite values (P5). MLIR spells infinities and NaN as raw hex
/// bit patterns, not as `inf`/`nan`, and guessing the spelling would produce a
/// module that either fails to parse or — worse — parses to the wrong number.
/// The causal mask in ARTX03 is what will need this, and it should arrive with
/// a test rather than as a silent fallback here.
pub fn emit_constant_splat(
    e: &mut MlirEmitter,
    value: f64,
    shape: &Shape,
) -> Result<SsaName, GlError> {
    let literal = mlir_scalar_literal(value, shape.dtype)?;
    let out = e.fresh();
    let t = shape.mlir_type();
    e.line(format!(
        r#"{out} = "stablehlo.constant"() {{value = dense<{literal}> : {t}}} : () -> {t}"#
    ));
    Ok(out)
}

/// Emits a rank-0 `stablehlo.constant`.
pub fn emit_constant_scalar(
    e: &mut MlirEmitter,
    value: f64,
    dtype: DType,
) -> Result<SsaName, GlError> {
    emit_constant_splat(e, value, &Shape::scalar(dtype))
}

/// Emits a dense `stablehlo.constant` from `f32` data, as a raw byte blob.
///
/// Used for the RoPE cos/sin tables and the causal mask, both of which carry
/// per-element values that no splat can express.
///
/// ⭐ The literal is `dense<"0x…">` — MLIR's raw-bytes form — rather than a
/// nested decimal list. Three reasons, in order of importance:
///
/// 1. **Exact.** No float → decimal text → float round trip, so the constant
///    the backend sees is bit-identical to the one Rust computed.
/// 2. **Handles ±∞ and NaN**, which have no decimal spelling at all. The causal
///    mask is made of −∞.
/// 3. Roughly 6× smaller than decimal text.
///
/// Byte order is the host's little-endian layout in C order, which is what
/// MLIR's parser expects; this was checked by round-tripping a tensor
/// containing −∞ through jaxlib's parser and comparing values, not just
/// checking that it parsed.
///
/// # Errors
/// If the data length disagrees with the shape, or the tensor exceeds
/// [`MAX_DENSE_CONSTANT_ELEMS`] — see that constant for why refusing is right.
pub fn emit_constant_dense_f32(
    e: &mut MlirEmitter,
    data: &[f32],
    shape: &Shape,
) -> Result<SsaName, GlError> {
    if shape.dtype != DType::F32 {
        return Err(GlError::UnsupportedDtype(format!(
            "emit_constant_dense_f32 called with a {:?} shape",
            shape.dtype
        )));
    }
    if data.len() != shape.numel() {
        return Err(GlError::ShapeMismatch {
            expected: shape.dims.clone(),
            got: vec![data.len()],
        });
    }
    if data.len() > MAX_DENSE_CONSTANT_ELEMS {
        return Err(GlError::Engine(format!(
            "dense constant of {} elements ({} MiB of MLIR text) exceeds the \
             {MAX_DENSE_CONSTANT_ELEMS}-element cap. Pass it as a runtime weight \
             instead of a constant — ARTX01 §7.2 allows either for the RoPE table, \
             and a mask this size wants the same treatment",
            data.len(),
            data.len() * 8 / (1024 * 1024),
        )));
    }

    let mut hex = String::with_capacity(data.len() * 8 + 2);
    for v in data {
        // Little-endian, C order — verified against MLIR's parser.
        for byte in v.to_le_bytes() {
            hex.push_str(&format!("{byte:02X}"));
        }
    }

    let out = e.fresh();
    let t = shape.mlir_type();
    e.line(format!(
        r#"{out} = "stablehlo.constant"() {{value = dense<"0x{hex}"> : {t}}} : () -> {t}"#
    ));
    Ok(out)
}

/// Element cap on a dense constant before [`emit_constant_dense_f32`] refuses.
///
/// ⚠️ A dense constant costs ~8 bytes of MLIR *text* per element. ARTX03 calls
/// a 512×512 causal mask "1 MB, acceptable for v1" — but the mask is O(S²), so
/// ARTX05's 2048 bucket is 4.2 M elements and **34 MB of text**, per bucket,
/// recompiled per shape. Silently emitting that is how a compile step turns
/// into a mystery hang.
///
/// 1 Mi elements ≈ 8 MB of text: comfortably past every test shape, well short
/// of the sizes that hurt. Refusing with a message beats guessing (P5).
pub const MAX_DENSE_CONSTANT_ELEMS: usize = 1 << 20;

/// Formats a single value as an MLIR literal of the given dtype.
///
/// ⚠️ Finite float literals go through `{:?}`, not `{}`. Rust's `Display`
/// prints `2.0f64` as `2`, and `dense<2> : tensor<f32>` is an integer literal
/// in a float attribute — an ambiguity that has no reason to exist in generated
/// IR. `Debug` always emits a decimal point.
///
/// Non-finite values have no decimal spelling and are emitted as the raw bit
/// pattern (`dense<0xFF800000> : tensor<f32>` is −∞). Softmax's reduce init and
/// the causal mask both need this.
fn mlir_scalar_literal(value: f64, dtype: DType) -> Result<String, GlError> {
    match dtype {
        DType::F64 | DType::F32 | DType::BF16 | DType::F16 => {
            if value.is_finite() {
                return Ok(mlir_float_literal(value));
            }
            non_finite_bits(value, dtype)
        }
        DType::I64 | DType::I32 | DType::I16 | DType::I8 => {
            if value.fract() != 0.0 || !value.is_finite() {
                return Err(GlError::UnsupportedDtype(format!(
                    "constant {value} is not representable as {dtype:?}"
                )));
            }
            Ok(format!("{}", value as i64))
        }
        DType::Bool => Ok(if value != 0.0 { "true" } else { "false" }.to_owned()),
    }
}

/// Formats a finite float as an MLIR float literal.
///
/// ⛔ **MLIR's lexer requires a decimal point in the mantissa.** Its float token
/// is `[0-9]+ '.' [0-9]* ([eE][-+]?[0-9]+)?` — so `1e-6` does not lex as a
/// float at all; it lexes as the integer `1` followed by garbage, and the
/// parser reports `expected '>'` several characters later.
///
/// Rust's `{:?}` produces `1e-6` for exactly the values that matter here:
/// `1e-6` is the RMSNorm epsilon of every Qwen2, Llama and Mistral model. So
/// every model gljax could trace emitted an unparseable module, and the
/// structural tests — which only assert `dense<1e-6>` is present — passed.
fn mlir_float_literal(value: f64) -> String {
    let s = format!("{value:?}");
    match s.find(['e', 'E']) {
        // `1e-6` -> `1.0e-6`
        Some(exp) if !s[..exp].contains('.') => {
            format!("{}.0{}", &s[..exp], &s[exp..])
        }
        Some(_) => s,
        // `2` -> `2.0`. Rust's Debug already does this for f64, but the
        // invariant is cheap to guarantee rather than rely on.
        None if !s.contains('.') => format!("{s}.0"),
        None => s,
    }
}

/// The MLIR hex spelling of ±∞ / NaN in a given float type.
///
/// Written out per type rather than derived, because the widths differ and a
/// pattern that is one nibble short parses as a *different finite number* —
/// silently.
fn non_finite_bits(value: f64, dtype: DType) -> Result<String, GlError> {
    let neg = value.is_sign_negative();
    let nan = value.is_nan();
    Ok(match dtype {
        DType::F64 => {
            let bits: u64 = if nan {
                0x7FF8_0000_0000_0000
            } else if neg {
                0xFFF0_0000_0000_0000
            } else {
                0x7FF0_0000_0000_0000
            };
            format!("0x{bits:016X}")
        }
        DType::F32 => {
            let bits: u32 = if nan {
                0x7FC0_0000
            } else if neg {
                0xFF80_0000
            } else {
                0x7F80_0000
            };
            format!("0x{bits:08X}")
        }
        // BF16 is the top 16 bits of the f32 pattern.
        DType::BF16 => {
            let bits: u16 = if nan {
                0x7FC0
            } else if neg {
                0xFF80
            } else {
                0x7F80
            };
            format!("0x{bits:04X}")
        }
        DType::F16 => {
            let bits: u16 = if nan {
                0x7E00
            } else if neg {
                0xFC00
            } else {
                0x7C00
            };
            format!("0x{bits:04X}")
        }
        other => {
            return Err(GlError::UnsupportedDtype(format!(
                "{value} is not representable as {other:?}"
            )))
        }
    })
}

// ---------------------------------------------------------------------------
// Shape ops
// ---------------------------------------------------------------------------

/// Formats a `usize` slice as an MLIR dense-array attribute.
///
/// ⛔ The empty case is **not** `array<i64: >`. MLIR's parser reads the colon
/// as introducing a value list and then reports `expected integer literal`;
/// the empty spelling is a bare `array<i64>`.
///
/// This is not hypothetical: broadcasting a rank-0 tensor — RMSNorm's epsilon,
/// softmax's mask, every scalar constant that has to meet a tensor — emits
/// exactly zero broadcast dimensions. Wave A2's structural tests asserted the
/// op was present and passed; jaxlib's MLIR parser rejected the module.
fn fmt_i64_array(v: &[usize]) -> String {
    if v.is_empty() {
        return "array<i64>".to_owned();
    }
    let mut s = String::from("array<i64: ");
    for (i, d) in v.iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        s.push_str(&d.to_string());
    }
    s.push('>');
    s
}

/// Formats a `usize` slice as an MLIR list literal `[a, b]`.
///
/// Unlike a dense array, the empty list `[]` is the correct spelling here —
/// which is why `dot_general`'s batching dimensions parse fine while an empty
/// `array<i64: >` does not.
fn fmt_list(v: &[usize]) -> String {
    let mut s = String::from("[");
    for (i, d) in v.iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        s.push_str(&d.to_string());
    }
    s.push(']');
    s
}

/// Emits `stablehlo.reshape`. No attributes — the output shape lives entirely
/// in the type signature.
pub fn emit_reshape(
    e: &mut MlirEmitter,
    operand: SsaName,
    in_shape: &Shape,
    out_shape: &Shape,
) -> SsaName {
    let out = e.fresh();
    e.line(format!(
        r#"{out} = "stablehlo.reshape"({operand}) : ({}) -> {}"#,
        in_shape.mlir_type(),
        out_shape.mlir_type()
    ));
    out
}

/// Emits `stablehlo.transpose`.
pub fn emit_transpose(
    e: &mut MlirEmitter,
    operand: SsaName,
    permutation: &[usize],
    in_shape: &Shape,
    out_shape: &Shape,
) -> SsaName {
    let out = e.fresh();
    e.line(format!(
        r#"{out} = "stablehlo.transpose"({operand}) {{permutation = {}}} : ({}) -> {}"#,
        fmt_i64_array(permutation),
        in_shape.mlir_type(),
        out_shape.mlir_type()
    ));
    out
}

/// Emits `stablehlo.slice` — static bounds, `[start:limit:stride]` per dim.
pub fn emit_slice(
    e: &mut MlirEmitter,
    operand: SsaName,
    starts: &[usize],
    limits: &[usize],
    strides: &[usize],
    in_shape: &Shape,
    out_shape: &Shape,
) -> SsaName {
    let out = e.fresh();
    e.line(format!(
        r#"{out} = "stablehlo.slice"({operand}) {{start_indices = {}, limit_indices = {}, strides = {}}} : ({}) -> {}"#,
        fmt_i64_array(starts),
        fmt_i64_array(limits),
        fmt_i64_array(strides),
        in_shape.mlir_type(),
        out_shape.mlir_type()
    ));
    out
}

/// Emits `stablehlo.concatenate`.
pub fn emit_concatenate(
    e: &mut MlirEmitter,
    operands: &[(SsaName, Shape)],
    dimension: usize,
    out_shape: &Shape,
) -> SsaName {
    let names: Vec<String> = operands.iter().map(|(n, _)| n.to_string()).collect();
    let types: Vec<String> = operands.iter().map(|(_, s)| s.mlir_type()).collect();
    let out = e.fresh();
    e.line(format!(
        r#"{out} = "stablehlo.concatenate"({}) {{dimension = {dimension} : i64}} : ({}) -> {}"#,
        names.join(", "),
        types.join(", "),
        out_shape.mlir_type()
    ));
    out
}

/// Emits `stablehlo.broadcast_in_dim`.
///
/// `broadcast_dims[i]` says which **output** dimension input dimension `i` maps
/// to. Getting this backwards produces a correctly-shaped, wrongly-populated
/// tensor — P4's bug class, which is why the direction is spelled out here and
/// tested in [`crate::graph`].
pub fn emit_broadcast_in_dim(
    e: &mut MlirEmitter,
    operand: SsaName,
    broadcast_dims: &[usize],
    in_shape: &Shape,
    out_shape: &Shape,
) -> SsaName {
    let out = e.fresh();
    e.line(format!(
        r#"{out} = "stablehlo.broadcast_in_dim"({operand}) {{broadcast_dimensions = {}}} : ({}) -> {}"#,
        fmt_i64_array(broadcast_dims),
        in_shape.mlir_type(),
        out_shape.mlir_type()
    ));
    out
}

// ---------------------------------------------------------------------------
// dot_general
// ---------------------------------------------------------------------------

/// Which dimensions of each operand are batched and which are contracted.
///
/// ⛔ This struct is where transposed matmuls come from. ARTX12 §B-T0 lists
/// "transposed dimension numbers" first among the structural bugs that produce
/// correct shapes and wrong numbers.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DotDimensionNumbers {
    pub lhs_batching: Vec<usize>,
    pub rhs_batching: Vec<usize>,
    pub lhs_contracting: Vec<usize>,
    pub rhs_contracting: Vec<usize>,
}

/// Emits `stablehlo.dot_general` — every matmul, projection, and attention
/// score product in the engine.
///
/// ⚠️ `precision_config` is hardcoded to `DEFAULT`. ARTX08 A8.α flags
/// `precision_config` / `preferred_element_type` plumbing as a **hard gate**
/// that blocks ARTX10 entirely, because quantization is a numerics contract.
/// This is where that plumbing will attach; until then, DEFAULT is what JAX
/// emits for an unannotated matmul, so gljax is not doing anything unusual.
#[allow(clippy::too_many_arguments)]
pub fn emit_dot_general(
    e: &mut MlirEmitter,
    lhs: SsaName,
    rhs: SsaName,
    dnums: &DotDimensionNumbers,
    lhs_shape: &Shape,
    rhs_shape: &Shape,
    out_shape: &Shape,
    numerics: crate::matrix::DotNumerics,
    accumulate: Option<DType>,
) -> SsaName {
    use crate::matrix::DotNumerics;

    let out = e.fresh();
    e.line(format!(r#"{out} = "stablehlo.dot_general"({lhs}, {rhs}) {{"#));
    e.push_indent();
    e.line("dot_dimension_numbers = #stablehlo.dot<");
    e.push_indent();
    e.line(format!(
        "lhs_batching_dimensions = {},",
        fmt_list(&dnums.lhs_batching)
    ));
    e.line(format!(
        "rhs_batching_dimensions = {},",
        fmt_list(&dnums.rhs_batching)
    ));
    e.line(format!(
        "lhs_contracting_dimensions = {},",
        fmt_list(&dnums.lhs_contracting)
    ));
    e.line(format!(
        "rhs_contracting_dimensions = {}",
        fmt_list(&dnums.rhs_contracting)
    ));
    e.pop_indent();
    e.line(">,");
    // `algorithm` and `precision_config` are mutually exclusive on
    // `stablehlo.dot_general` — exactly one of these arms fires.
    let numerics_trailer = if accumulate.is_some() { "," } else { "" };
    match numerics {
        DotNumerics::Default => e.line(format!(
            "precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]{numerics_trailer}"
        )),
        DotNumerics::Highest => e.line(format!(
            "precision_config = [#stablehlo<precision HIGHEST>, #stablehlo<precision HIGHEST>]{numerics_trailer}"
        )),
        DotNumerics::Algorithm(alg) => {
            e.line(format!("algorithm = {}{numerics_trailer}", alg.mlir_str()))
        }
    }
    if let Some(acc) = accumulate {
        e.line(format!("preferred_element_type = {}", acc.mlir_str()));
    }
    e.pop_indent();
    e.line(format!(
        r#"}} : ({}, {}) -> {}"#,
        lhs_shape.mlir_type(),
        rhs_shape.mlir_type(),
        out_shape.mlir_type()
    ));
    out
}

// ---------------------------------------------------------------------------
// gather
// ---------------------------------------------------------------------------

/// How a `stablehlo.gather` maps indices onto the operand.
///
/// ⛔ Six interacting index lists, and a wrong one produces a correctly-shaped
/// tensor of the wrong rows. For the embedding lookup that means every token
/// silently embeds as some other token.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GatherDimensionNumbers {
    /// Output dimensions that carry the sliced window.
    pub offset_dims: Vec<usize>,
    /// Operand dimensions the slice is size-1 in, dropped from the output.
    pub collapsed_slice_dims: Vec<usize>,
    /// Operand batching dimensions (newer spec field; empty for a plain lookup).
    pub operand_batching_dims: Vec<usize>,
    /// Index batching dimensions (newer spec field; empty for a plain lookup).
    pub start_indices_batching_dims: Vec<usize>,
    /// Which operand dimension each index component addresses.
    pub start_index_map: Vec<usize>,
    /// Which dimension of `start_indices` holds the index vector.
    pub index_vector_dim: usize,
}

/// Emits `stablehlo.gather` — embedding lookup, and later KV reads.
// Eight arguments because `gather` genuinely has eight independent inputs: two
// operands, six dimension numbers, the slice sizes, and three types the emitter
// cannot infer. Bundling them into a struct would only move the list.
#[allow(clippy::too_many_arguments)]
pub fn emit_gather(
    e: &mut MlirEmitter,
    operand: SsaName,
    start_indices: SsaName,
    dnums: &GatherDimensionNumbers,
    slice_sizes: &[usize],
    operand_shape: &Shape,
    indices_shape: &Shape,
    out_shape: &Shape,
) -> SsaName {
    let out = e.fresh();
    e.line(format!(
        r#"{out} = "stablehlo.gather"({operand}, {start_indices}) {{"#
    ));
    e.push_indent();
    e.line("dimension_numbers = #stablehlo.gather<");
    e.push_indent();
    e.line(format!("offset_dims = {},", fmt_list(&dnums.offset_dims)));
    e.line(format!(
        "collapsed_slice_dims = {},",
        fmt_list(&dnums.collapsed_slice_dims)
    ));
    e.line(format!(
        "operand_batching_dims = {},",
        fmt_list(&dnums.operand_batching_dims)
    ));
    e.line(format!(
        "start_indices_batching_dims = {},",
        fmt_list(&dnums.start_indices_batching_dims)
    ));
    e.line(format!(
        "start_index_map = {},",
        fmt_list(&dnums.start_index_map)
    ));
    e.line(format!("index_vector_dim = {}", dnums.index_vector_dim));
    e.pop_indent();
    e.line(">,");
    e.line(format!("slice_sizes = {},", fmt_i64_array(slice_sizes)));
    e.line("indices_are_sorted = false");
    e.pop_indent();
    e.line(format!(
        r#"}} : ({}, {}) -> {}"#,
        operand_shape.mlir_type(),
        indices_shape.mlir_type(),
        out_shape.mlir_type()
    ));
    out
}

// ---------------------------------------------------------------------------
// dynamic slice — the KV cache primitives (ARTX05)
// ---------------------------------------------------------------------------

/// Emits `stablehlo.dynamic_slice`.
///
/// Unlike [`emit_slice`], the start offsets are **runtime values** — rank-0
/// integer tensors, one per dimension — while the sizes stay static. That is
/// exactly the shape of a decode step: the position varies, the amount read
/// does not.
///
/// ⚠️ Out-of-range starts are **clamped**, not an error. StableHLO's spec says
/// the effective start is `clamp(0, start, dim_size − slice_size)`, so reading
/// past the end of a KV cache silently returns the last valid window instead of
/// failing. Nothing downstream can tell the difference — P4 — so the caller
/// must bound the position itself.
pub fn emit_dynamic_slice(
    e: &mut MlirEmitter,
    operand: SsaName,
    start_indices: &[(SsaName, Shape)],
    slice_sizes: &[usize],
    operand_shape: &Shape,
    out_shape: &Shape,
) -> SsaName {
    let out = e.fresh();
    let mut operands = String::from(&operand.to_string());
    let mut types = String::from(&operand_shape.mlir_type());
    for (name, shape) in start_indices {
        operands.push_str(&format!(", {name}"));
        types.push_str(&format!(", {}", shape.mlir_type()));
    }
    e.line(format!(
        r#"{out} = "stablehlo.dynamic_slice"({operands}) {{slice_sizes = {}}} : ({types}) -> {}"#,
        fmt_i64_array(slice_sizes),
        out_shape.mlir_type()
    ));
    out
}

/// Emits `stablehlo.dynamic_update_slice` — the KV cache write.
///
/// Returns a *new* tensor equal to `operand` with `update` written at the
/// runtime offsets. ARTX05's scatter-on-write: the functional form is what lets
/// XLA turn it into an in-place store when the input buffer is donated, so
/// there is no separate mutation op to reach for.
///
/// Same clamping caveat as [`emit_dynamic_slice`].
pub fn emit_dynamic_update_slice(
    e: &mut MlirEmitter,
    operand: SsaName,
    update: SsaName,
    start_indices: &[(SsaName, Shape)],
    operand_shape: &Shape,
    update_shape: &Shape,
) -> SsaName {
    let out = e.fresh();
    let mut operands = format!("{operand}, {update}");
    let mut types = format!(
        "{}, {}",
        operand_shape.mlir_type(),
        update_shape.mlir_type()
    );
    for (name, shape) in start_indices {
        operands.push_str(&format!(", {name}"));
        types.push_str(&format!(", {}", shape.mlir_type()));
    }
    e.line(format!(
        r#"{out} = "stablehlo.dynamic_update_slice"({operands}) : ({types}) -> {}"#,
        operand_shape.mlir_type()
    ));
    out
}

// ---------------------------------------------------------------------------
// reduce
// ---------------------------------------------------------------------------

/// Emits `stablehlo.reduce` with a caller-supplied combiner region.
///
/// ⛔ **Deviation from ARTX02 §3.** The sketch there emits the region as
/// `(` … `)` and the attributes as `) {dimensions = ...}`. The spec's form
/// wraps the region in braces:
///
/// ```text
/// %r = "stablehlo.reduce"(%in, %init) ({
///   ^bb0(%a: tensor<f32>, %b: tensor<f32>):
///     ...
///     "stablehlo.return"(%c) : (tensor<f32>) -> ()
/// }) {dimensions = array<i64: 1>} : (tensor<1x6xf32>, tensor<f32>) -> tensor<1xf32>
/// ```
///
/// Without the `{` … `}` the module does not parse. This matters more than it
/// looks: `reduce` is inside RMSNorm and softmax, so every layer of every
/// model would have carried it.
///
/// Region details the spec is strict about: the block label is `^bb0`, region
/// arguments are **rank-0 tensors** of the element type (`tensor<f32>`, never
/// bare `f32`), and the terminator is the quoted `"stablehlo.return"`.
pub fn emit_reduce(
    e: &mut MlirEmitter,
    operand: SsaName,
    init: SsaName,
    reduce_dims: &[usize],
    in_shape: &Shape,
    out_shape: &Shape,
    combiner: impl FnOnce(&mut MlirEmitter, SsaName, SsaName) -> SsaName,
) -> SsaName {
    let scalar = Shape::scalar(in_shape.dtype);
    let st = scalar.mlir_type();

    let out = e.fresh();
    let arg_a = e.fresh();
    let arg_b = e.fresh();

    e.line(format!(
        r#"{out} = "stablehlo.reduce"({operand}, {init}) ({{"#
    ));
    e.push_indent();
    e.line(format!("^bb0({arg_a}: {st}, {arg_b}: {st}):"));
    e.push_indent();
    let result = combiner(e, arg_a, arg_b);
    e.line(format!(
        r#""stablehlo.return"({result}) : ({st}) -> ()"#
    ));
    e.pop_indent();
    e.pop_indent();
    e.line(format!(
        r#"}}) {{dimensions = {}}} : ({}, {st}) -> {}"#,
        fmt_i64_array(reduce_dims),
        in_shape.mlir_type(),
        out_shape.mlir_type()
    ));
    out
}

/// `reduce` with an `add` combiner — the sum in RMSNorm and softmax.
pub fn emit_reduce_add(
    e: &mut MlirEmitter,
    operand: SsaName,
    zero: SsaName,
    reduce_dims: &[usize],
    in_shape: &Shape,
    out_shape: &Shape,
) -> SsaName {
    let dtype = in_shape.dtype;
    emit_reduce(
        e,
        operand,
        zero,
        reduce_dims,
        in_shape,
        out_shape,
        move |inner, a, b| emit_add(inner, a, b, &Shape::scalar(dtype)),
    )
}

/// `reduce` with a `maximum` combiner — the subtract-max step of a
/// numerically stable softmax.
pub fn emit_reduce_max(
    e: &mut MlirEmitter,
    operand: SsaName,
    neg_inf: SsaName,
    reduce_dims: &[usize],
    in_shape: &Shape,
    out_shape: &Shape,
) -> SsaName {
    let dtype = in_shape.dtype;
    emit_reduce(
        e,
        operand,
        neg_inf,
        reduce_dims,
        in_shape,
        out_shape,
        move |inner, a, b| emit_maximum(inner, a, b, &Shape::scalar(dtype)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_emits_generic_form_with_both_operand_types() {
        let mut e = MlirEmitter::new();
        let a = e.fresh();
        let b = e.fresh();
        let ty = Shape::scalar(DType::F32);
        let out = emit_add(&mut e, a, b, &ty);
        assert_eq!(out, SsaName(2));
        assert_eq!(
            e.into_body(),
            "  %v2 = \"stablehlo.add\"(%v0, %v1) : (tensor<f32>, tensor<f32>) -> tensor<f32>\n"
        );
    }

    #[test]
    fn exponential_is_not_spelled_exp() {
        // The spec mnemonic is `exponential`; `stablehlo.exp` does not exist.
        let mut e = MlirEmitter::new();
        let x = e.fresh();
        emit_exponential(&mut e, x, &Shape::new([4], DType::F32));
        assert!(e.body().contains(r#""stablehlo.exponential""#), "{}", e.body());
    }

    /// ⛔ Regression: MLIR's lexer needs a decimal point in the mantissa.
    /// `dense<1e-6>` — the RMSNorm epsilon of every Llama-family model — does
    /// not parse; caught by jaxlib's parser, missed by the structural tests.
    #[test]
    fn exponent_notation_keeps_a_decimal_point_in_the_mantissa() {
        assert_eq!(mlir_float_literal(1e-6), "1.0e-6");
        assert_eq!(mlir_float_literal(-1e-6), "-1.0e-6");
        assert_eq!(mlir_float_literal(1.5e-6), "1.5e-6");
        assert_eq!(mlir_float_literal(2.0), "2.0");
        assert_eq!(mlir_float_literal(0.125), "0.125");

        let mut e = MlirEmitter::new();
        emit_constant_scalar(&mut e, 1e-6, DType::F32).expect("eps");
        assert!(e.body().contains("dense<1.0e-6>"), "{}", e.body());
    }

    #[test]
    fn float_constants_keep_a_decimal_point() {
        // `format!("{}", 2.0f64)` yields "2", which is an integer literal in
        // MLIR. The whole point of using Debug here.
        let mut e = MlirEmitter::new();
        emit_constant_scalar(&mut e, 2.0, DType::F32).expect("finite constant");
        assert!(
            e.body().contains("dense<2.0>"),
            "expected a decimal point, got: {}",
            e.body()
        );
    }

    #[test]
    fn splat_constant_carries_the_full_shape() {
        let mut e = MlirEmitter::new();
        let shape = Shape::new([2, 3], DType::F32);
        emit_constant_splat(&mut e, 0.5, &shape).expect("finite constant");
        assert_eq!(
            e.into_body(),
            "  %v0 = \"stablehlo.constant\"() {value = dense<0.5> : tensor<2x3xf32>} : () -> tensor<2x3xf32>\n"
        );
    }

    /// −∞ is softmax's reduce init and the causal mask's fill value, so it has
    /// to have an exact spelling. The width is per-type and a short pattern
    /// parses as a *finite* number rather than failing.
    #[test]
    fn non_finite_constants_emit_the_right_width_bit_pattern() {
        let cases = [
            (DType::F32, f64::NEG_INFINITY, "dense<0xFF800000>"),
            (DType::F32, f64::INFINITY, "dense<0x7F800000>"),
            (DType::F64, f64::NEG_INFINITY, "dense<0xFFF0000000000000>"),
            (DType::BF16, f64::NEG_INFINITY, "dense<0xFF80>"),
            (DType::F16, f64::NEG_INFINITY, "dense<0xFC00>"),
        ];
        for (dtype, value, expected) in cases {
            let mut e = MlirEmitter::new();
            emit_constant_scalar(&mut e, value, dtype).expect("non-finite constant");
            assert!(
                e.body().contains(expected),
                "{dtype:?} {value}: expected {expected}, got {}",
                e.body()
            );
        }
    }

    #[test]
    fn non_finite_integer_constants_are_still_refused() {
        let mut e = MlirEmitter::new();
        let err = emit_constant_splat(&mut e, f64::NEG_INFINITY, &Shape::scalar(DType::I32))
            .expect_err("-inf is not an i32");
        assert!(matches!(err, GlError::UnsupportedDtype(_)), "{err:?}");
    }

    #[test]
    fn dense_f32_constants_emit_little_endian_raw_bytes() {
        // 1.0f32 is 0x3F800000; little-endian that is 00 00 80 3F. Verified
        // against jaxlib's MLIR parser, values compared not just parsed.
        let mut e = MlirEmitter::new();
        emit_constant_dense_f32(
            &mut e,
            &[1.0, f32::NEG_INFINITY],
            &Shape::new([2], DType::F32),
        )
        .expect("dense constant");
        assert!(
            e.body().contains(r#"dense<"0x0000803F000080FF">"#),
            "{}",
            e.body()
        );
    }

    #[test]
    fn dense_constants_refuse_a_length_that_disagrees_with_the_shape() {
        let mut e = MlirEmitter::new();
        let err = emit_constant_dense_f32(&mut e, &[1.0, 2.0], &Shape::new([3], DType::F32))
            .expect_err("2 values is not a [3] tensor");
        assert!(matches!(err, GlError::ShapeMismatch { .. }), "{err:?}");
    }

    #[test]
    fn oversized_dense_constants_are_refused_with_a_way_forward() {
        // A 2048-bucket causal mask is 4.2M elements — 34 MB of MLIR text.
        let mut e = MlirEmitter::new();
        let n = MAX_DENSE_CONSTANT_ELEMS + 1;
        let err = emit_constant_dense_f32(&mut e, &vec![0.0; n], &Shape::new([n], DType::F32))
            .expect_err("must refuse rather than emit tens of MB");
        let msg = err.to_string();
        assert!(msg.contains("runtime weight"), "{msg}");
    }

    #[test]
    fn gather_emits_every_dimension_number_field() {
        let mut e = MlirEmitter::new();
        let table = e.fresh();
        let idx = e.fresh();
        let dnums = GatherDimensionNumbers {
            offset_dims: vec![1],
            collapsed_slice_dims: vec![0],
            start_index_map: vec![0],
            index_vector_dim: 1,
            ..Default::default()
        };
        emit_gather(
            &mut e,
            table,
            idx,
            &dnums,
            &[1, 896],
            &Shape::new([151936, 896], DType::F32),
            &Shape::new([128, 1], DType::I32),
            &Shape::new([128, 896], DType::F32),
        );
        let body = e.into_body();
        assert!(body.contains("offset_dims = [1],"), "{body}");
        assert!(body.contains("collapsed_slice_dims = [0],"), "{body}");
        assert!(body.contains("operand_batching_dims = [],"), "{body}");
        assert!(body.contains("start_index_map = [0],"), "{body}");
        assert!(body.contains("index_vector_dim = 1"), "{body}");
        assert!(body.contains("slice_sizes = array<i64: 1, 896>,"), "{body}");
    }

    #[test]
    fn fractional_values_are_refused_for_integer_dtypes() {
        let mut e = MlirEmitter::new();
        let err = emit_constant_splat(&mut e, 1.5, &Shape::scalar(DType::I32))
            .expect_err("1.5 is not an i32");
        assert!(matches!(err, GlError::UnsupportedDtype(_)), "{err:?}");
    }

    #[test]
    fn bool_constants_spell_true_and_false() {
        let mut e = MlirEmitter::new();
        emit_constant_scalar(&mut e, 1.0, DType::Bool).expect("bool constant");
        emit_constant_scalar(&mut e, 0.0, DType::Bool).expect("bool constant");
        assert!(e.body().contains("dense<true> : tensor<i1>"));
        assert!(e.body().contains("dense<false> : tensor<i1>"));
    }

    #[test]
    fn convert_states_both_the_source_and_target_type() {
        let mut e = MlirEmitter::new();
        let x = e.fresh();
        let bf16 = Shape::new([8, 16], DType::BF16);
        let f32 = Shape::new([8, 16], DType::F32);
        emit_convert(&mut e, x, &bf16, &f32);
        assert_eq!(
            e.into_body(),
            "  %v1 = \"stablehlo.convert\"(%v0) : (tensor<8x16xbf16>) -> tensor<8x16xf32>\n"
        );
    }

    #[test]
    fn slice_emits_all_three_static_bound_arrays() {
        let mut e = MlirEmitter::new();
        let x = e.fresh();
        emit_slice(
            &mut e,
            x,
            &[1, 2],
            &[3, 4],
            &[1, 1],
            &Shape::new([3, 4], DType::I64),
            &Shape::new([2, 2], DType::I64),
        );
        let body = e.into_body();
        assert!(body.contains("start_indices = array<i64: 1, 2>"), "{body}");
        assert!(body.contains("limit_indices = array<i64: 3, 4>"), "{body}");
        assert!(body.contains("strides = array<i64: 1, 1>"), "{body}");
    }

    #[test]
    fn transpose_emits_the_permutation_in_order() {
        let mut e = MlirEmitter::new();
        let x = e.fresh();
        emit_transpose(
            &mut e,
            x,
            &[0, 2, 1, 3],
            &Shape::new([1, 512, 16, 128], DType::BF16),
            &Shape::new([1, 16, 512, 128], DType::BF16),
        );
        let body = e.into_body();
        assert!(
            body.contains("permutation = array<i64: 0, 2, 1, 3>"),
            "{body}"
        );
        assert!(
            body.contains("(tensor<1x512x16x128xbf16>) -> tensor<1x16x512x128xbf16>"),
            "{body}"
        );
    }

    #[test]
    fn concatenate_lists_every_operand_and_its_type() {
        let mut e = MlirEmitter::new();
        let a = e.fresh();
        let b = e.fresh();
        let half = Shape::new([2, 4], DType::F32);
        emit_concatenate(
            &mut e,
            &[(a, half.clone()), (b, half)],
            1,
            &Shape::new([2, 8], DType::F32),
        );
        assert_eq!(
            e.into_body(),
            "  %v2 = \"stablehlo.concatenate\"(%v0, %v1) {dimension = 1 : i64} : \
             (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x8xf32>\n"
        );
    }

    #[test]
    fn dot_general_emits_all_four_dimension_number_lists() {
        let mut e = MlirEmitter::new();
        let a = e.fresh();
        let b = e.fresh();
        let dnums = DotDimensionNumbers {
            lhs_batching: vec![],
            rhs_batching: vec![],
            lhs_contracting: vec![1],
            rhs_contracting: vec![0],
        };
        emit_dot_general(
            &mut e,
            a,
            b,
            &dnums,
            &Shape::new([4, 8], DType::F32),
            &Shape::new([8, 16], DType::F32),
            &Shape::new([4, 16], DType::F32),
            crate::matrix::DotNumerics::Default,
            None,
        );
        let body = e.into_body();
        assert!(body.contains("lhs_batching_dimensions = [],"), "{body}");
        assert!(body.contains("rhs_batching_dimensions = [],"), "{body}");
        assert!(body.contains("lhs_contracting_dimensions = [1],"), "{body}");
        assert!(body.contains("rhs_contracting_dimensions = [0]"), "{body}");
        assert!(
            body.contains("precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]"),
            "{body}"
        );
        assert!(
            body.contains("} : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>"),
            "{body}"
        );
    }

    /// ⛔ Regression test for the ARTX02 §3 sketch, which emitted the region
    /// without its braces. `(` instead of `({` does not parse, and `reduce` is
    /// inside both RMSNorm and softmax — every layer of every model.
    #[test]
    fn reduce_wraps_its_region_in_braces() {
        let mut e = MlirEmitter::new();
        let x = e.fresh();
        let zero = e.fresh();
        emit_reduce_add(
            &mut e,
            x,
            zero,
            &[1],
            &Shape::new([1, 6], DType::F32),
            &Shape::new([1], DType::F32),
        );
        let body = e.into_body();
        assert!(
            body.contains(r#""stablehlo.reduce"(%v0, %v1) ({"#),
            "region must open with `({{`:\n{body}"
        );
        assert!(
            body.contains(r#"}) {dimensions = array<i64: 1>}"#),
            "region must close with `}})` before the attributes:\n{body}"
        );
    }

    #[test]
    fn reduce_region_args_are_rank_zero_tensors_not_bare_scalars() {
        // The spec is explicit: `^bb0(%a: tensor<f32>, ...)`, never `%a: f32`.
        let mut e = MlirEmitter::new();
        let x = e.fresh();
        let zero = e.fresh();
        emit_reduce_add(
            &mut e,
            x,
            zero,
            &[2],
            &Shape::new([1, 512, 2048], DType::F32),
            &Shape::new([1, 512], DType::F32),
        );
        let body = e.into_body();
        assert!(
            body.contains("^bb0(%v3: tensor<f32>, %v4: tensor<f32>):"),
            "{body}"
        );
        assert!(
            body.contains(r#""stablehlo.return"(%v5) : (tensor<f32>) -> ()"#),
            "{body}"
        );
        // The init operand's type appears in the signature as a rank-0 tensor.
        assert!(
            body.contains("(tensor<1x512x2048xf32>, tensor<f32>) -> tensor<1x512xf32>"),
            "{body}"
        );
    }

    #[test]
    fn reduce_max_uses_a_maximum_combiner() {
        let mut e = MlirEmitter::new();
        let x = e.fresh();
        let init = e.fresh();
        emit_reduce_max(
            &mut e,
            x,
            init,
            &[1],
            &Shape::new([4, 8], DType::F32),
            &Shape::new([4], DType::F32),
        );
        assert!(e.body().contains(r#""stablehlo.maximum""#), "{}", e.body());
    }
}
