//! Numerics knobs for `stablehlo.dot_general` (ARTX08 §"A8.α — the numerics
//! plumbing fix").
//!
//! ⛔ **Scope note.** ARTX08 as a whole specifies a much larger matrix-compute
//! layer — structural fusion, MoE `ragged_dot`, kernel-lowering choke points
//! (Waves A8.1–A8.5). None of that is built here. This module is exactly
//! Wave A8.α: the two numeric controls `stablehlo.dot_general` already
//! supports in the spec but that gljax could not previously reach, because
//! `emit_dot_general` hardcoded `precision_config = [DEFAULT, DEFAULT]` and
//! `infer_dot_general_shape` always took the LHS dtype as the output dtype
//! regardless of what the backend was told to accumulate in.
//!
//! ARTX08 calls this "the single highest-value change in this document":
//! without it, a BF16 model's accumulation type is "probably fine, decided by
//! the backend, unstatable by us" — exactly the wrong property for a project
//! whose ARTX01 built an F64 oracle to validate numerics against.

use crate::stablehlo::types::DType;

/// Numeric contract for a single contraction.
///
/// `Default` reproduces gljax's behavior before this module existed, byte for
/// byte: `precision_config = [DEFAULT, DEFAULT]`, no `algorithm` attribute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DotNumerics {
    /// `precision_config = [DEFAULT, DEFAULT]`. What gljax emitted before A8.α.
    #[default]
    Default,
    /// `precision_config = [HIGHEST, HIGHEST]`. Backend emulates a wider dtype
    /// than the operands' storage type, without naming which algorithm it uses.
    Highest,
    /// An explicit, named numeric contract. Mutually exclusive with
    /// `precision_config` per the StableHLO spec — `emit_dot_general` enforces
    /// this by construction (the two are different match arms, never both
    /// emitted).
    Algorithm(DotAlgorithm),
}

/// A named `stablehlo.dot_algorithm` preset.
///
/// This is a curated subset of the presets XLA recognizes (there are also
/// F16/BF16_BF16_BF16/float8 presets) — only the ones ARTX08 identifies as
/// relevant to gljax's CPU/GPU bring-up: bf16 production numerics, the `_X3`/
/// `_X6` emulation passes for when bf16 alone is too imprecise, tf32 for
/// Ampere-class GPUs, and the f32/f64 oracle path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DotAlgorithm {
    /// bf16 x bf16 -> f32. Standard production numerics.
    Bf16Bf16F32,
    /// bf16 x bf16 -> f32, 3-pass emulation of higher precision.
    Bf16Bf16F32X3,
    /// bf16 x bf16 -> f32, 6-pass emulation of higher precision.
    Bf16Bf16F32X6,
    /// TensorFloat32 x TensorFloat32 -> f32. Ampere+ GPU only.
    Tf32Tf32F32,
    /// tf32 x tf32 -> f32, 3-pass emulation.
    Tf32Tf32F32X3,
    /// f32 x f32 -> f32. No precision reduction.
    F32F32F32,
    /// f64 x f64 -> f64. ARTX01 §3.4's correctness oracle.
    F64F64F64,
}

impl DotAlgorithm {
    /// The `lhs_precision_type`/`rhs_precision_type`/`accumulation_type`
    /// triple, plus component/pass counts, per
    /// `stablehlo/dialect/StablehloAttrs.td`'s `DotAlgorithm` attribute.
    ///
    /// `allow_imprecise_accumulation` is `false` for every preset here — that
    /// flag exists for the float8 "fast accum" presets, none of which are in
    /// this enum.
    const fn fields(self) -> (&'static str, &'static str, &'static str, i64) {
        // (lhs_precision_type, rhs_precision_type, accumulation_type, num_primitive_operations)
        // lhs_component_count and rhs_component_count are always 1 for these
        // presets — decomposition into multiple components (not multiple
        // passes) is a float8 concern, per JAX's DotAlgorithmPreset docs.
        match self {
            DotAlgorithm::Bf16Bf16F32 => ("bf16", "bf16", "f32", 1),
            DotAlgorithm::Bf16Bf16F32X3 => ("bf16", "bf16", "f32", 3),
            DotAlgorithm::Bf16Bf16F32X6 => ("bf16", "bf16", "f32", 6),
            DotAlgorithm::Tf32Tf32F32 => ("tf32", "tf32", "f32", 1),
            DotAlgorithm::Tf32Tf32F32X3 => ("tf32", "tf32", "f32", 3),
            DotAlgorithm::F32F32F32 => ("f32", "f32", "f32", 1),
            DotAlgorithm::F64F64F64 => ("f64", "f64", "f64", 1),
        }
    }

    /// The MLIR attribute body: `dot_algorithm<lhs_precision_type = ..., ...>`.
    ///
    /// Verified against `stablehlo/dialect/StablehloAttrs.td`'s assembly
    /// format directly (field names and order), not guessed.
    pub fn mlir_str(self) -> String {
        let (lhs, rhs, acc, num_ops) = self.fields();
        format!(
            "#stablehlo.dot_algorithm<lhs_precision_type = {lhs}, rhs_precision_type = {rhs}, \
             accumulation_type = {acc}, lhs_component_count = 1, rhs_component_count = 1, \
             num_primitive_operations = {num_ops}, allow_imprecise_accumulation = false>"
        )
    }
}

/// Options for a single `dot_general`. All fields default to gljax's
/// pre-A8.α behavior: `MatmulOpts::default()` emits byte-identical MLIR to
/// what every existing call site emitted before this module existed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MatmulOpts {
    pub numerics: DotNumerics,
    /// Recommended accumulation dtype (`preferred_element_type`). `None`
    /// leaves the output dtype as the (already-reconciled) operand dtype —
    /// today's behavior.
    pub accumulate: Option<DType>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_numerics_is_the_default_variant() {
        assert_eq!(DotNumerics::default(), DotNumerics::Default);
    }

    #[test]
    fn default_matmul_opts_has_no_accumulate_override() {
        let opts = MatmulOpts::default();
        assert_eq!(opts.numerics, DotNumerics::Default);
        assert_eq!(opts.accumulate, None);
    }

    #[test]
    fn bf16_x3_mlir_str_has_three_primitive_operations() {
        let s = DotAlgorithm::Bf16Bf16F32X3.mlir_str();
        assert!(s.contains("lhs_precision_type = bf16"), "{s}");
        assert!(s.contains("rhs_precision_type = bf16"), "{s}");
        assert!(s.contains("accumulation_type = f32"), "{s}");
        assert!(s.contains("num_primitive_operations = 3"), "{s}");
        assert!(s.contains("allow_imprecise_accumulation = false"), "{s}");
    }

    #[test]
    fn f64_oracle_algorithm_is_f64_throughout() {
        let s = DotAlgorithm::F64F64F64.mlir_str();
        assert!(s.contains("lhs_precision_type = f64"), "{s}");
        assert!(s.contains("rhs_precision_type = f64"), "{s}");
        assert!(s.contains("accumulation_type = f64"), "{s}");
        assert!(s.contains("num_primitive_operations = 1"), "{s}");
    }

    #[test]
    fn every_preset_has_a_distinct_mlir_string() {
        let all = [
            DotAlgorithm::Bf16Bf16F32,
            DotAlgorithm::Bf16Bf16F32X3,
            DotAlgorithm::Bf16Bf16F32X6,
            DotAlgorithm::Tf32Tf32F32,
            DotAlgorithm::Tf32Tf32F32X3,
            DotAlgorithm::F32F32F32,
            DotAlgorithm::F64F64F64,
        ];
        let mut strs: Vec<String> = all.iter().map(|a| a.mlir_str()).collect();
        strs.sort();
        strs.dedup();
        assert_eq!(strs.len(), all.len(), "two presets collided on their MLIR spelling");
    }
}
