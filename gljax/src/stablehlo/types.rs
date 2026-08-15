//! MLIR type formatting — `DType`, `Shape`, `ParamKind` (ARTX02 §2).

/// Element types gljax can name in StableHLO.
///
/// ARTX02 §2 notes the spec's canonical integer spelling is `si32`/`ui32`, but
/// emits `i32`/`i64` because that is what JAX and PyTorch emit and what PJRT
/// parses. gljax follows the same choice.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F64,
    F32,
    BF16,
    F16,
    I64,
    I32,
    I16,
    I8,
    /// `i1` in MLIR.
    Bool,
}

impl DType {
    /// The MLIR spelling.
    pub const fn mlir_str(self) -> &'static str {
        match self {
            DType::F64 => "f64",
            DType::F32 => "f32",
            DType::BF16 => "bf16",
            DType::F16 => "f16",
            DType::I64 => "i64",
            DType::I32 => "i32",
            DType::I16 => "i16",
            DType::I8 => "i8",
            DType::Bool => "i1",
        }
    }

    /// Bytes per element in a dense host buffer.
    pub const fn byte_size(self) -> usize {
        match self {
            DType::F64 | DType::I64 => 8,
            DType::F32 | DType::I32 => 4,
            DType::BF16 | DType::F16 | DType::I16 => 2,
            DType::I8 | DType::Bool => 1,
        }
    }
}

/// A static tensor shape: dims plus element type.
///
/// ⚠️ **Static dims only.** P3 makes every distinct shape a separate compiled
/// artifact; there is no `?` dimension in gljax v1 (ARTX02 §2, ARTX01 §9.3).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Shape {
    pub dims: Vec<usize>,
    pub dtype: DType,
}

impl Shape {
    pub fn new(dims: impl Into<Vec<usize>>, dtype: DType) -> Self {
        Shape {
            dims: dims.into(),
            dtype,
        }
    }

    /// A 0-dimensional tensor. StableHLO has no first-class scalars — the
    /// spelling is `tensor<f32>`, a tensor of rank 0.
    pub fn scalar(dtype: DType) -> Self {
        Shape {
            dims: Vec::new(),
            dtype,
        }
    }

    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }

    pub fn byte_len(&self) -> usize {
        self.numel() * self.dtype.byte_size()
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// The MLIR `TensorType` spelling: `tensor<2x4xf32>`, or `tensor<f32>` at
    /// rank 0.
    pub fn mlir_type(&self) -> String {
        if self.dims.is_empty() {
            return format!("tensor<{}>", self.dtype.mlir_str());
        }
        let mut s = String::from("tensor<");
        for d in &self.dims {
            s.push_str(&d.to_string());
            s.push('x');
        }
        s.push_str(self.dtype.mlir_str());
        s.push('>');
        s
    }
}

/// Whether a `func.func` parameter carries activations, weights, or a KV cache.
///
/// The distinction matters at runtime, not in the emitted type: ARTX01 §8.2's
/// compile/weight separation means weights are uploaded once and reused across
/// calls, while inputs change every call. `KvCache` is ARTX05's third kind — a
/// resident buffer that, unlike a weight, is written every decode step, which
/// is why it is the only kind [`FuncBuilder::alias_output`] permits donating.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParamKind {
    Input,
    Weight,
    KvCache,
}

/// A declared `func.func` parameter: what it is called, what shape it has, and
/// which SSA name it was bound to.
///
/// ⭐ `name` is the load-bearing field for weights. ARTX02 §6 makes the scope
/// stack produce checkpoint keys verbatim — `model.layers.0.self_attn.q_proj.weight`
/// is both the traced name and the safetensors key, so the loader in ARTX04
/// needs no translation table. A translation table is a thing that can drift;
/// a name that is already correct cannot.
#[derive(Clone, Debug)]
pub struct ParamDesc {
    pub name: String,
    pub shape: Shape,
    pub kind: ParamKind,
    pub ssa: crate::stablehlo::emitter::SsaName,
    /// ARTX05 §6 buffer donation: `Some(i)` means this parameter aliases
    /// output `i` — XLA gives the output the same buffer as the input instead
    /// of copying. Only ever set on [`ParamKind::KvCache`] params; see
    /// [`FuncBuilder::alias_output`](crate::graph::builder::FuncBuilder::alias_output).
    pub alias_output: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_zero_shape_formats_without_dimensions() {
        // StableHLO spells a scalar as a rank-0 tensor. `tensor<xf32>` — the
        // obvious result of a naive join — does not parse.
        assert_eq!(Shape::scalar(DType::F32).mlir_type(), "tensor<f32>");
        assert_eq!(Shape::scalar(DType::F32).numel(), 1);
        assert_eq!(Shape::scalar(DType::F32).rank(), 0);
    }

    #[test]
    fn ranked_shapes_format_as_dim_x_dim_x_dtype() {
        assert_eq!(
            Shape::new([2, 4], DType::F32).mlir_type(),
            "tensor<2x4xf32>"
        );
        assert_eq!(Shape::new([128], DType::BF16).mlir_type(), "tensor<128xbf16>");
        assert_eq!(
            Shape::new([151936, 896], DType::BF16).mlir_type(),
            "tensor<151936x896xbf16>"
        );
    }

    #[test]
    fn byte_len_multiplies_element_count_by_element_size() {
        // Qwen2-0.5B hidden dim, the shape most of the sprint's tensors carry.
        let s = Shape::new([128, 896], DType::BF16);
        assert_eq!(s.numel(), 114_688);
        assert_eq!(s.byte_len(), 229_376);
    }

    #[test]
    fn bool_is_spelled_i1() {
        assert_eq!(DType::Bool.mlir_str(), "i1");
        assert_eq!(DType::Bool.byte_size(), 1);
    }
}
